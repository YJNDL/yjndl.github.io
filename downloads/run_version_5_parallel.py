from __future__ import annotations

import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for thin envs
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKERS = 14
DEFAULT_REPRESENTATIVE_ROOT = str(SCRIPT_DIR / "materials_ecp_representatives")
DEFAULT_VERSION_SCRIPT = str(SCRIPT_DIR / "version_5.0.py")
DEFAULT_RUN_PYTHON = os.environ.get("BOND_DENSITY_PYTHON", sys.executable)
DEFAULT_RUN_ROOT = str(SCRIPT_DIR / "ecp_runs")
THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(slots=True)
class RunConfig:
    representative_root: str = DEFAULT_REPRESENTATIVE_ROOT
    version_script: str = DEFAULT_VERSION_SCRIPT
    python_executable: str = DEFAULT_RUN_PYTHON
    run_root: str = DEFAULT_RUN_ROOT
    run_name: str | None = None
    parallel_workers: int = DEFAULT_WORKERS
    space_groups: list[str] = field(default_factory=list)
    scan_mode: str = "auto"
    step_size: float = 0.4
    xrd_min_intensity: float = 0.1
    xrd_two_theta_min: float = 0.0
    xrd_two_theta_max: float = 90.0
    auto_generate_xrd: bool = True
    deduplicate_identical_slabs: bool = True
    limit: int = 0
    prepare_only: bool = False
    resume: bool = False


# =========================
# User Config
# 直接修改这一段，然后运行本脚本
# 如果在服务器上运行，建议把本脚本和 version_5.0.py
# 放在同一个目录下，例如 /public/home/yt1/zyl
# =========================
USER_CONFIG = RunConfig(
    representative_root=DEFAULT_REPRESENTATIVE_ROOT,
    version_script=DEFAULT_VERSION_SCRIPT,
    python_executable=DEFAULT_RUN_PYTHON,
    run_root=DEFAULT_RUN_ROOT,
    run_name=None,
    parallel_workers=16,
    space_groups=[],
    scan_mode="auto",
    step_size=0.4,
    xrd_min_intensity=0.1,
    xrd_two_theta_min=0.0,
    xrd_two_theta_max=90.0,
    auto_generate_xrd=True,
    deduplicate_identical_slabs=True,
    limit=0,
    prepare_only=False,
    resume=False,
)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_manifest_rows(manifest_csv: Path) -> list[dict]:
    with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_representative_path_from_manifest(row: dict, representative_root: Path) -> Path:
    representative_file = row.get("representative_file", "").strip()
    if not representative_file:
        representative_file = Path(row.get("source_representative_path", "")).name
    if not representative_file:
        representative_file = Path(row.get("ecp_representative_path", "")).name

    spacegroup_dir = row.get("spacegroup_dir", "").strip()
    type_dir_name = row.get("type_dir_name", "").strip()
    if spacegroup_dir and type_dir_name and representative_file:
        candidate = representative_root / spacegroup_dir / type_dir_name / representative_file
        if candidate.exists():
            return candidate

    for key in ("ecp_representative_path", "source_representative_path"):
        raw_path = row.get(key, "").strip()
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate

    missing_current_path = representative_root / spacegroup_dir / type_dir_name / representative_file
    raise FileNotFoundError(
        "Representative file not found in current representative root or manifest path: "
        f"{missing_current_path}"
    )


def collect_input_rows(representative_root: Path) -> list[dict]:
    manifest_csv = representative_root / "summary" / "ecp_representative_manifest.csv"
    if manifest_csv.exists():
        rows = load_manifest_rows(manifest_csv)
        return [
            {
                "spacegroup_number": row["spacegroup_number"],
                "spacegroup_symbol": row["spacegroup_symbol"],
                "spacegroup_dir": row["spacegroup_dir"],
                "structure_type_id": row["structure_type_id"],
                "type_dir_name": row["type_dir_name"],
                "member_count": row["member_count"],
                "source_representative_path": str(resolve_representative_path_from_manifest(row, representative_root)),
                "file_name": row.get("representative_file", "").strip()
                or Path(row.get("source_representative_path", "")).name,
            }
            for row in rows
        ]

    rows = []
    for path in sorted(representative_root.rglob("mp-*.vasp")):
        type_dir = path.parent
        sg_dir = type_dir.parent.name if type_dir.parent != representative_root else ""
        rows.append(
            {
                "spacegroup_number": "",
                "spacegroup_symbol": "",
                "spacegroup_dir": sg_dir,
                "structure_type_id": type_dir.name,
                "type_dir_name": type_dir.name,
                "member_count": "",
                "source_representative_path": str(path),
                "file_name": path.name,
            }
        )
    return rows


def normalize_space_group_request(token: str) -> tuple[str, str]:
    text = token.strip()
    if not text:
        raise ValueError("Empty space-group selector is not allowed.")

    if text.isdigit():
        return ("number", str(int(text)))

    normalized = text.upper()
    if normalized.startswith("SG_"):
        parts = normalized.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return ("number", str(int(parts[1])))
        return ("dir", normalized)

    return ("dir", normalized)


def filter_rows_by_space_groups(rows: list[dict], requested_space_groups: list[str]) -> list[dict]:
    if not requested_space_groups:
        return rows

    requested_numbers = set()
    requested_dirs = set()
    for token in requested_space_groups:
        kind, value = normalize_space_group_request(token)
        if kind == "number":
            requested_numbers.add(value)
        else:
            requested_dirs.add(value)

    filtered_rows = []
    for row in rows:
        row_number = row.get("spacegroup_number", "").strip()
        row_number = str(int(row_number)) if row_number else ""
        row_dir = row.get("spacegroup_dir", "").strip().upper()
        if row_number in requested_numbers or row_dir in requested_dirs:
            filtered_rows.append(row)
    return filtered_rows


def ensure_unique_file_names(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for row in rows:
        file_name = row["file_name"]
        source_path = row["source_representative_path"]
        if file_name in seen and seen[file_name] != source_path:
            duplicates.append(file_name)
        seen[file_name] = source_path
    if duplicates:
        duplicate_preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Found duplicate representative file names that would collide in flat input staging: {duplicate_preview}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stage_inputs(rows: list[dict], flat_input_dir: Path, resume: bool) -> list[dict]:
    staged_rows = []
    iterator = rows if tqdm is None else tqdm(rows, desc="Staging representatives")
    for index, row in enumerate(iterator, start=1):
        source_path = Path(row["source_representative_path"]).resolve()
        staged_path = flat_input_dir / source_path.name
        if not staged_path.exists():
            hardlink_or_copy(source_path, staged_path)
        elif not resume:
            raise FileExistsError(f"Staged input already exists: {staged_path}")
        staged_rows.append(
            {
                **row,
                "staged_input_path": str(staged_path),
            }
        )
        if tqdm is None and (index % 1000 == 0 or index == len(rows)):
            print(f"Staged representatives: {index}/{len(rows)}", flush=True)
    return staged_rows


def build_run_name(prefix: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}"


def validate_resume_selection(existing_manifest_path: Path, requested_rows: list[dict]) -> None:
    existing_rows = load_manifest_rows(existing_manifest_path)
    existing_sources = {row["source_representative_path"] for row in existing_rows}
    requested_sources = {row["source_representative_path"] for row in requested_rows}
    if existing_sources != requested_sources:
        raise ValueError(
            "Resume selection does not match the existing run manifest. "
            "Use the same filters and limit, or choose a new --run-name."
        )


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in THREAD_LIMIT_ENV.items():
        env.setdefault(key, value)
    return env


def main(config: RunConfig | None = None) -> int:
    config = config or USER_CONFIG

    representative_root = Path(config.representative_root).resolve()
    version_script = Path(config.version_script).resolve()
    python_executable = Path(os.path.abspath(os.path.expanduser(config.python_executable)))
    run_root = Path(config.run_root).resolve()

    if not representative_root.exists():
        print(f"Representative root not found: {representative_root}", file=sys.stderr)
        return 1
    if not version_script.exists():
        print(f"version_5.0.py not found: {version_script}", file=sys.stderr)
        return 1
    if not python_executable.exists():
        print(f"Python executable not found: {python_executable}", file=sys.stderr)
        return 1

    rows = collect_input_rows(representative_root)
    rows = filter_rows_by_space_groups(rows, config.space_groups or [])
    if config.limit > 0:
        rows = rows[: config.limit]
    if not rows:
        print("No representative structures found.", file=sys.stderr)
        return 1

    ensure_unique_file_names(rows)

    run_name = config.run_name or build_run_name("bond_density_run")
    run_dir = run_root / run_name
    if run_dir.exists() and not config.resume:
        print(f"Run directory already exists: {run_dir}", file=sys.stderr)
        return 1

    flat_input_dir = run_dir / "inputs_flat"
    xrd_dir = run_dir / "XRD"
    run_root.mkdir(parents=True, exist_ok=True)
    flat_input_dir.mkdir(parents=True, exist_ok=True)
    xrd_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "prepared_inputs_manifest.csv"
    if config.resume and manifest_path.exists():
        try:
            validate_resume_selection(manifest_path, rows)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    staged_rows = stage_inputs(rows, flat_input_dir, resume=config.resume)
    write_csv(
        manifest_path,
        staged_rows,
        [
            "spacegroup_number",
            "spacegroup_symbol",
            "spacegroup_dir",
            "structure_type_id",
            "type_dir_name",
            "member_count",
            "source_representative_path",
            "file_name",
            "staged_input_path",
        ],
    )

    print(f"Prepared inputs: {len(staged_rows)}")
    print(f"Selected space groups: {len({row['spacegroup_dir'] for row in staged_rows})}")
    print(f"Run directory: {run_dir}")
    print(f"Flat input dir: {flat_input_dir}")
    print(f"XRD dir: {xrd_dir}")
    print("Thread limits: " + ", ".join(f"{key}={value}" for key, value in THREAD_LIMIT_ENV.items()))

    command = [
        str(python_executable),
        str(version_script),
        "batch",
        "--poscar-folder",
        str(flat_input_dir),
        "--excel-folder",
        str(xrd_dir),
        "--parallel-workers",
        str(max(1, config.parallel_workers)),
        "--scan-mode",
        config.scan_mode,
        "--step-size",
        str(config.step_size),
        "--xrd-min-intensity",
        str(config.xrd_min_intensity),
        "--xrd-two-theta-min",
        str(config.xrd_two_theta_min),
        "--xrd-two-theta-max",
        str(config.xrd_two_theta_max),
    ]
    command.append("--auto-generate-xrd" if config.auto_generate_xrd else "--no-auto-generate-xrd")
    command.append(
        "--deduplicate-identical-slabs"
        if config.deduplicate_identical_slabs
        else "--no-deduplicate-identical-slabs"
    )

    command_preview = " ".join(command)
    (run_dir / "run_command.txt").write_text(command_preview + "\n", encoding="utf-8")
    print("Launch command:")
    print(command_preview)

    if config.prepare_only:
        print("prepare-only mode: version_5.0.py was not launched.")
        return 0

    completed = subprocess.run(command, cwd=run_dir, env=build_subprocess_env())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
