#!/bin/bash
set -euo pipefail

# 待处理的 mp-*.vasp 所在目录
POSCAR_FOLDER="./structures—exp"

# 生成作业目录的根目录
JOB_ROOT="."

# 要复制进去的主脚本和提交脚本模板
SCRIPT_TEMPLATE="./version_4.0.py"
NEWVASP_TEMPLATE="./newvasp.sh"

# 若本地已经有 XRD 表，可按材料名复制到每个作业目录的 XRD/ 下
XRD_SOURCE_DIR="./XRD"
COPY_XRD_IF_EXISTS=1

mkdir -p "$JOB_ROOT"

prepared_count=0
shopt -s nullglob
for poscar_path in "$POSCAR_FOLDER"/mp-*.vasp; do
  poscar_name=$(basename "$poscar_path")
  material_name="${poscar_name%.vasp}"
  job_dir="$JOB_ROOT/$material_name"
  xrd_dir="$job_dir/XRD"

  mkdir -p "$xrd_dir"

  cp "$poscar_path" "$job_dir/$poscar_name"
  cp "$SCRIPT_TEMPLATE" "$job_dir/version_4.0.py"
  cp "$NEWVASP_TEMPLATE" "$job_dir/newvasp.sh"
  chmod +x "$job_dir/newvasp.sh"

  if [[ "$COPY_XRD_IF_EXISTS" == "1" ]]; then
    xrd_file_name="${material_name}-XRD.xlsx"
    if [[ -f "$XRD_SOURCE_DIR/$xrd_file_name" ]]; then
      cp "$XRD_SOURCE_DIR/$xrd_file_name" "$xrd_dir/$xrd_file_name"
    fi
  fi

  # 让每个目录里的 version_4.0.py 能直接用 `python version_4.0.py` 运行。
  sed -i.bak "s|^        'poscar_folder': .*|        'poscar_folder': '.',|" "$job_dir/version_4.0.py"
  sed -i.bak "s|^        'excel_folder': .*|        'excel_folder': 'XRD',|" "$job_dir/version_4.0.py"
  sed -i.bak "s|^        'poscar_name': .*|        'poscar_name': '${poscar_name}',|" "$job_dir/version_4.0.py"
  sed -i.bak "s|^        'parallel_workers': .*|        'parallel_workers': 1,|" "$job_dir/version_4.0.py"
  rm -f "$job_dir/version_4.0.py.bak"

  prepared_count=$((prepared_count + 1))
done

echo "prepared_count=$prepared_count"
echo "job_root=$JOB_ROOT"
