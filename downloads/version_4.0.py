### Contributors: Zhong Yongle, Lin Kangyang, Xiao Yuxiang, Du Yingning
### V1.0 实现对晶胞的切面，求面积，求面间距离，但是只能实现对alpha = bata = 90°的晶胞进行操作，且晶面指数必须为正数 ### 2023.09.26
### V1.2 根据晶体学知识完成对晶面指数的处理 ### 2023.09.28
### V1.5 采用Barycentric Coordinates解决点是否在晶体内部，对任意晶系有效 ### 2023.09.29
### V2.0 完成输入输出文件的大循环，引入进度条 ### 2023.09.30
### V2.3 排序方法成功完善 ### 2023.10.3
### V2.7 完成对边界原子的处理 ### 2023.10.5
### V2.9 完成根据空间群算法解决判断成键的问题 ### 2023.10.11
### V3.0 基本功能完善 ###
### V4.0 稳定版 slab 逻辑 + StructureGraph/jimage 判键 + V_old + 等价 hkl 展开 ###

"""
配置区说明
========

1. 直接运行 `python version_4.0.py` 时，会读取脚本顶部的 `SCRIPT_RUN_CONFIG`。
2. `mode='batch'` 表示完整跑剥离分析。
3. `mode='generate-xrd'` 表示只生成 XRD 表。
4. 如果你在命令行手动追加了 `batch` 或 `generate-xrd` 参数，则命令行参数优先。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from ase.io import read
from tqdm import tqdm


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

PLANE_TOLERANCE = 1e-6
DEFAULT_MIN_VACUUM_SIZE = 5.0
DEFAULT_SLAB_REPEAT_COUNT = 3
DEFAULT_PARALLEL_WORKERS = max(1, (os.cpu_count() or 1) // 2)
SLAB_STRUCTURE_MATCHER = StructureMatcher(
    ltol=1e-4,
    stol=1e-4,
    angle_tol=0.1,
    primitive_cell=False,
    scale=False,
    attempt_supercell=False,
)

# =========================
# 直接运行脚本时的配置区
# 修改这里即可控制输入参数
# 你平时只需要改这一个字典。
# 如果直接运行 `python version_4.0.py`，程序会自动读取这里的设置。
# 如果手动追加命令行参数，则命令行参数优先。
# =========================
SCRIPT_RUN_CONFIG = {
    # mode:
    # 'batch'        -> 完整跑剥离分析
    # 'generate-xrd' -> 只生成 XRD 表
    'mode': 'batch',

    # batch 模式配置
    'batch': {
        # poscar_folder:
        # 待分析的 mp-*.vasp 所在目录
        'poscar_folder': '/Users/zhongyongle/PycharmProjects/PythonProject/structures—exp',

        # excel_folder:
        # XRD Excel 表所在目录；若缺表且 auto_generate_xrd=True，会自动在这里生成
        'excel_folder': '/Users/zhongyongle/PycharmProjects/PythonProject/XRD',

        # poscar_name:
        # 若为 None，则扫描 poscar_folder 下全部 mp-*.vasp
        # 若填写具体文件名，例如 'mp-23501-FeBiO3.vasp'，则只处理这个材料
        'poscar_name': None,

        # step_size:
        # fixed 模式下的切面扫描步长，单位为 Angstrom；越小越细，但运行越慢
        'step_size': 0.4,

        # scan_mode:
        # 'auto'  -> 自动比较 fixed / event 的扫描点数量，选择更省的一种
        # 'event' -> 基于关键事件点扫描，通常更精确
        # 'fixed' -> 传统等距步长扫描，作为回退模式保留
        'scan_mode': 'auto',

        # auto_generate_xrd:
        # True  -> 缺少 XRD.xlsx 时自动生成
        # False -> 如果缺少 XRD.xlsx，则直接跳过该材料
        'auto_generate_xrd': True,

        # xrd_two_theta_min / xrd_two_theta_max:
        # 自动生成 XRD 时使用的 2theta 范围
        'xrd_two_theta_min': 0.0,
        'xrd_two_theta_max': 90.0,

        # xrd_min_intensity:
        # batch 真正扫描前采用的强度过滤阈值
        # 0    -> 不过滤，保留所有 hkl
        # 5.0  -> 只保留强度 >= 5 的 hkl
        # 10.0 -> 过滤更严格，运行更快
        'xrd_min_intensity': 15.0,

        # parallel_workers:
        # 并行处理的进程数。
        # 1 表示串行；大于 1 表示按“不同材料”并行。
        # 默认取 CPU 核心数的一半，通常更稳。
        'parallel_workers': DEFAULT_PARALLEL_WORKERS,

        # deduplicate_identical_slabs:
        # True  -> 先生成每个 hkl 对应的 slab，再比较 slab 结构；若完全相同，则复用代表面的扫描结果
        # False -> 不做这一步 slab 级去重，所有 hkl 都单独扫描
        'deduplicate_identical_slabs': True,
    },

    # generate-xrd 模式配置
    'generate_xrd': {
        # input_dir:
        # 输入结构目录，会递归扫描其中所有 .vasp 文件
        'input_dir': '/Users/zhongyongle/Desktop/ABC3_空间群_结构类型分类',

        # output_dir:
        # XRD Excel 输出目录
        'output_dir': '/Users/zhongyongle/Desktop/zyl/code/ECP/XRD',

        # two_theta_min / two_theta_max:
        # 生成 XRD 表时保留的 2theta 范围
        'two_theta_min': 0.0,
        'two_theta_max': 90.0,

        # min_intensity:
        # 写入 XRD.xlsx 的最低强度阈值
        # 0 -> 不过滤
        'min_intensity': 5.0,
    },
}

CONFIG_HELP = """
默认情况下，直接运行 `python version_4.0.py` 会读取脚本顶部的 `SCRIPT_RUN_CONFIG`。
如果手动追加了 `batch` 或 `generate-xrd` 命令行参数，则命令行参数优先。
"""


def get_site_symbol(site):
    """兼容普通元素和带氧化态的物种对象，统一取元素符号。"""
    try:
        return site.specie.symbol
    except Exception:
        first_species = next(iter(site.species))
        return first_species.symbol


def point_to_plane_distance(point, plane):
    """计算点到平面的有符号距离，正负号表示位于平面两侧。"""
    numerator = np.dot(plane[:3], point) + plane[3]
    denominator = np.linalg.norm(plane[:3])
    return numerator / denominator


def prepare_structure_for_bonding(structure):
    """
    在判键前尽量补上氧化态信息。

    CrystalNN 在离子晶体里通常会受氧化态影响；若自动猜测失败，则退回中性结构继续计算。
    """
    bond_structure = structure.copy()
    oxidation_guessed = False
    try:
        bond_structure.add_oxidation_state_by_guess()
        oxidation_guessed = True
    except Exception:
        pass
    return bond_structure, oxidation_guessed


def get_normal_repeat_period(structure, miller_index, vacuum_size=DEFAULT_MIN_VACUUM_SIZE):
    """
    计算指定 hkl 方向上的法向真实重复周期 L_repeat。

    这里取的是 SlabGenerator 构造出的 oriented unit cell 在 c 方向上的长度。
    它代表结构沿 slab 法向完整重复一次所需的距离，通常比 d_hkl 更适合用来定义
    “忽略表面 termination 的内部扫描窗口”。
    """
    probe_generator = SlabGenerator(
        structure,
        miller_index,
        center_slab=True,
        min_slab_size=1.0,
        min_vacuum_size=vacuum_size,
    )
    return float(probe_generator.oriented_unit_cell.lattice.c)


def build_center_scan_window(zmin, zmax, repeat_period, step_size):
    """
    根据 slab 的实际原子厚度，在中间构造一个宽度为 L_repeat 的扫描窗口。

    设计原则：
    1. 只扫 slab 中央区域，尽量压低上下表面 termination 的影响；
    2. 窗口宽度优先取一个完整法向重复周期；
    3. 若 slab 实际厚度不足一个周期，则退回为扫描整个原子厚度，并至少保留一个中心采样点。
    """
    slab_thickness = max(0.0, float(zmax - zmin))
    slab_center = float((zmin + zmax) / 2)
    window_width = min(float(repeat_period), slab_thickness)
    window_start = slab_center - window_width / 2
    window_end = slab_center + window_width / 2

    if window_width <= PLANE_TOLERANCE:
        heights = np.array([slab_center], dtype=float)
    else:
        heights = np.arange(window_start + step_size / 2, window_end, step_size, dtype=float)
        if len(heights) == 0:
            heights = np.array([slab_center], dtype=float)

    return {
        'slab_thickness': slab_thickness,
        'slab_center': slab_center,
        'window_width': window_width,
        'window_start': window_start,
        'window_end': window_end,
        'heights': heights,
    }


def build_old_volume_distance_samples(structure):
    """
    构造与 3.2 一致的距离采样点。

    老版本在估算“夹层体积”时，并不是只看原胞中的原子，而是会在 x/y 方向补一层周期镜像，
    然后再按点到切平面的距离找最近原子。由于当前切平面始终平行于 slab 表面，
    这些镜像不会改变 z 坐标，但会保留 3.2 里“边界原子会重复参与最近距离统计”的行为。
    """
    distance_points = []

    def mirror_coord(value):
        if value < 0.5:
            return value + 1
        if value > 0.5:
            return value - 1
        return value

    for site in structure:
        x, y, z = site.frac_coords
        image_fracs = [(x, y, z)]
        x_mirror = mirror_coord(x)
        y_mirror = mirror_coord(y)

        if x != x_mirror:
            image_fracs.append((x_mirror, y, z))
        if y != y_mirror:
            image_fracs.append((x, y_mirror, z))
        if x != x_mirror and y != y_mirror:
            image_fracs.append((x_mirror, y_mirror, z))

        for frac_coords in image_fracs:
            distance_points.append(structure.lattice.get_cartesian_coords(frac_coords))

    return distance_points


def compute_old_volume_from_plane(distance_points, plane, area):
    """
    用 3.2 的最近原子距离规则估算 V_old。

    规则保持与旧版一致：
    1. 分别找切平面上下两侧距离最近的两个原子；
    2. 两侧各自取最近两原子的平均距离；
    3. 两侧厚度相加得到 t_old；
    4. V_old = A * t_old。
    """
    if area == 0:
        return 0.0, 0.0

    signed_distances = [point_to_plane_distance(point, plane) for point in distance_points]
    positive_distances = sorted([distance for distance in signed_distances if distance >= 0], key=abs)
    negative_distances = sorted([distance for distance in signed_distances if distance < 0], key=abs)

    if len(positive_distances) >= 2:
        positive_mean = (positive_distances[0] + positive_distances[1]) / 2
    elif len(positive_distances) == 1:
        positive_mean = positive_distances[0] / 2
    else:
        positive_mean = 0.0

    if len(negative_distances) >= 2:
        negative_mean = (abs(negative_distances[0]) + abs(negative_distances[1])) / 2
    elif len(negative_distances) == 1:
        negative_mean = abs(negative_distances[0]) / 2
    else:
        negative_mean = 0.0

    mean_distances = float(positive_mean + negative_mean)
    mean_volume = float(mean_distances * area)
    return mean_volume, mean_distances


def convert_diffraction_hkl_to_three_index(raw_hkl):
    """
    把 XRDCalculator 给出的 hkl 统一转成后续 slab 生成使用的三指数。

    对三方/六方晶系里常见的四指数 (h, k, i, l)，这里保留 (h, k, l)。
    """
    raw_hkl = tuple(int(value) for value in raw_hkl)
    if len(raw_hkl) == 4:
        return raw_hkl[0], raw_hkl[1], raw_hkl[3]
    return raw_hkl


def canonicalize_global_inversion(hkl):
    """
    把 (h, k, l) 与 (-h, -k, -l) 视为同一组，仅保留一个代表。

    注意这里只合并全局反号，不合并 102 与 012 这种由点群旋转得到的不同方向。
    """
    if all(value == 0 for value in hkl):
        return hkl

    for value in hkl:
        if value > 0:
            return hkl
        if value < 0:
            return tuple(-item for item in hkl)
    return hkl


def reduce_hkl_by_common_factor(hkl):
    """
    若 hkl 可以提取公因数，则化成最简整数比。

    例如：
    - (0, 0, 2) -> (0, 0, 1)
    - (2, 2, 0) -> (1, 1, 0)
    - (2, -4, 2) -> (1, -2, 1)
    """
    values = [abs(int(value)) for value in hkl if int(value) != 0]
    if not values:
        return tuple(int(value) for value in hkl)

    divisor = values[0]
    for value in values[1:]:
        divisor = math.gcd(divisor, value)

    if divisor <= 1:
        return tuple(int(value) for value in hkl)
    return tuple(int(value) // divisor for value in hkl)


def canonicalize_plane_direction(hkl):
    """
    把晶面方向统一成“去公因数 + 合并全局反号”的标准形式。

    这样 001/002 会视为同一种晶面，但 012/102 这种不同方向不会被错误合并。
    """
    reduced_hkl = reduce_hkl_by_common_factor(hkl)
    return canonicalize_global_inversion(reduced_hkl)


def format_hkl_label(hkl):
    """把 hkl 元组格式化成便于记录的文本。"""
    return f'({int(hkl[0])}, {int(hkl[1])}, {int(hkl[2])})'


def expand_equivalent_hkls(representative_hkl, point_group_ops):
    """
    展开一个代表 hkl 的所有对称等价方向。

    这里不再像之前那样在 XRD 阶段就把整个衍射家族压成一个代表面，否则后续 slab
    生成时会把 102 / 012 这类在程序实现中表现不同的方向错误地合并。

    参数 point_group_ops 是 SpacegroupAnalyzer 给出的点群操作列表。

    仍然保留最小限度的去重：
    - 合并完全相同的 hkl
    - 合并全局反号 (h, k, l) 与 (-h, -k, -l)
    """
    expanded_hkls = set()
    source_vector = np.array(representative_hkl, dtype=float)
    for operation in point_group_ops:
        mapped_vector = operation.rotation_matrix.dot(source_vector)
        mapped_hkl = tuple(int(round(value)) for value in mapped_vector)
        canonical_hkl = canonicalize_global_inversion(mapped_hkl)
        if canonical_hkl == (0, 0, 0):
            continue
        expanded_hkls.add(canonical_hkl)

    return sorted(expanded_hkls)

class PoscarProcessor:

    def __init__(self, poscar_file):
        """
        初始化PoscarProcessor类的实例。

        Args:
            poscar_file (str): POSCAR文件的路径。
        """
        self.poscar_file = poscar_file

    def parse_POSCAR(self):
        """
        解析POSCAR文件以获取晶格参数和原子坐标。

        Returns:
            tuple: (scaling_factor, lattice_vectors, atom_coordinates)
        """
        structure = Structure.from_file(self.poscar_file)
        lattice_vectors = np.array(structure.lattice.matrix)
        species_counts = {}
        atom_coordinates = []

        for index, site in enumerate(structure, start=1):
            symbol = get_site_symbol(site)
            species_counts[symbol] = species_counts.get(symbol, 0) + 1
            label_atom = f"{symbol}{species_counts[symbol]}"
            atom_coordinates.append(list(site.frac_coords) + [label_atom, index])

        return 1.0, lattice_vectors, atom_coordinates

    def calculate_vertices_from_poscar(self):
        """
        从POSCAR文件中计算顶点坐标。

        Returns:
            list: 顶点坐标列表。
        """
        structure = Structure.from_file(self.poscar_file)
        lattice_matrix = structure.lattice.matrix

        vertices = []
        for i in range(8):
            vertex_frac = [i // 4, (i % 4) // 2, (i % 2)]
            vertex_cart = structure.lattice.get_cartesian_coords(vertex_frac)
            vertices.append(vertex_cart)

        return vertices

    def process_poscar_file(self):
        """
        返回切面扫描所需的几何数据。

        Returns:
            tuple: (vertices, lattice_vectors, atom_coordinates)
        """
        scaling_factor, lattice_vectors, atom_data = self.parse_POSCAR()
        lattice_vectors *= scaling_factor
        structure = Structure.from_file(self.poscar_file)
        atom_coordinates = []
        for index, site in enumerate(structure, start=1):
            symbol = get_site_symbol(site)
            atom_coordinates.append(list(site.coords) + [f'{symbol}{index}', index])
        vertices = self.calculate_vertices_from_poscar()
        return vertices, lattice_vectors, atom_coordinates

def calculate_plane_equation(normal_vector, point_on_plane):
    '''Calculate plane equations'''

    a, b, c = normal_vector
    x0, y0, z0 = point_on_plane

    d = -(a*x0 + b*y0 + c*z0)

    return a, b, c, d

def compute_3d_polygon_area(points):
    '''Calculate cross-sectional area'''

    if (len(points) < 3): return 0.0
    P1X,P1Y,P1Z = points[0][0],points[0][1],points[0][2]
    P2X,P2Y,P2Z = points[1][0],points[1][1],points[1][2]
    P3X,P3Y,P3Z = points[2][0],points[2][1],points[2][2]
    a = pow(((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)),2) + pow(((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z)),2) + pow(((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)),2)
    cosnx = ((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z))/(pow(a,1/2))
    cosny = ((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z))/(pow(a,1/2))
    cosnz = ((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y))/(pow(a,1/2))
    s = cosnz*((points[-1][0])*(P1Y)-(P1X)*(points[-1][1])) + cosnx*((points[-1][1])*(P1Z)-(P1Y)*(points[-1][2])) + cosny*((points[-1][2])*(P1X)-(P1Z)*(points[-1][0]))
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        ss = cosnz*((p1[0])*(p2[1])-(p2[0])*(p1[1])) + cosnx*((p1[1])*(p2[2])-(p2[1])*(p1[2])) + cosny*((p1[2])*(p2[0])-(p2[2])*(p1[0]))
        s += ss

    s = abs(s/2.0)

    return s

def filter_and_sort_points(intersection_points, normal_vector):
    """
    对晶胞与切平面的交点去重并排序，得到可用于面积计算的闭合多边形顶点顺序。

    参数:
    - intersection_points: 原始交点的列表
    - normal_vector: 当前切平面的法向

    返回:
    - filtered_points: 经过筛选和排序的交点列表
    """
    filtered_points = []
    visited_points = set()
    centerX, centerY, centerZ = 0, 0, 0


    for point in intersection_points:
        x, y, z = point
        # 检查点是否已经存在于已访问的点集合中
        duplicate_found = False
        for visited_point in visited_points:
            if abs(point[0] - visited_point[0]) < 1e-4 and \
                    abs(point[1] - visited_point[1]) < 1e-4 and \
                    abs(point[2] - visited_point[2]) < 1e-4:
                duplicate_found = True
                break
        if not duplicate_found:
            
            centerX += point[0]
            centerY += point[1]
            centerZ += point[2]

            filtered_points.append(point)
            visited_points.add(point)
    if not filtered_points:
        return None

    centerX /= len(filtered_points)
    centerY /= len(filtered_points)
    centerZ /= len(filtered_points)

    delta_points = []
    up_and_right_points = []
    up_and_left_points = []
    down_and_right_points = []
    down_and_left_points = []
    for point in filtered_points:
        x, y, z = point
        delta_points.append([x - centerX, y - centerY, z - centerZ])
    vectorX=np.array(delta_points[0])
    vectorY=np.cross(vectorX,normal_vector)
    matrix=np.array([vectorX,vectorY])
    
    matrix=np.linalg.pinv(matrix.T)
    for point in delta_points:
        target=np.array(point)
        temp=matrix@ target        
        flag=True
        if temp[0] > 0:

            if temp[1] > 0:
                k=temp[1]/temp[0]
                for value in up_and_right_points:
                    if abs(value[1]-k)<1e-4:
                        flag=False
                        break
                
                if flag:
                    up_and_right_points.append([point,k])
            else:
                k=temp[1]/temp[0]
                for value in down_and_right_points:
                    if abs(value[1]-k)<1e-4:
                        flag=False
                        break
                
                if flag:
                    down_and_right_points.append([point,k])
        else:
            if temp[1] > 0:
                k=0
                if temp[0]!=0:
                    k=temp[1]/temp[0]
                for value in up_and_left_points:
                    if abs(value[1]-k)<1e-4:
                        flag=False
                        break
                
                if flag:
                    up_and_left_points.append([point,k])
            else:
                k=0
                if temp[0]!=0:
                    k=temp[1]/temp[0]
                for value in down_and_left_points:
                    if abs(value[1]-k)<1e-4:
                        flag=False
                        break
                
                if flag:
                    down_and_left_points.append([point,k])
    up_and_right_points.sort(key=lambda x: x[1], reverse=True)
    up_and_left_points.sort(key=lambda x: x[1],reverse= True)
    down_and_right_points.sort(key=lambda x: x[1], reverse=True)
    down_and_left_points.sort(key=lambda x: x[1],reverse=True)




    filtered_points = []
    filtered_points = \
    [i[0] for i in up_and_left_points] +[i[0] for i in up_and_right_points] + \
        [i[0] for i in down_and_right_points] + \
        [i[0] for i in down_and_left_points]
        

    for point in filtered_points:
        point[0] += centerX
        point[1] += centerY
        point[2] += centerZ
    # plot_structure_with_cube(poscar_file, filtered_points)
    return filtered_points

def compute_intersection_points(vertices, lattice_vectors, plane):
    """
    计算晶胞边与当前切平面的交点。

    参数:
    - vertices: 立方体的顶点列表
    - lattice_vectors: 3x3矩阵，每行是一个方向向量
    - plane: 代表晶面的方程的参数

    返回:
    - intersection_points: 与晶面交叉的立方体边上的点列表
    """

    def line_equation(point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2

        # 计算方向向量
        direction_vector = (x2 - x1, y2 - y1, z2 - z1)

        return point1, direction_vector

    def line_plane_intersection(line_start, line_direction, plane_coefficients):
        x0, y0, z0 = line_start
        a, b, c = line_direction
        A, B, C, D = plane_coefficients

        # 计算连线方程与平面的交点
        t = (-A * x0 - B * y0 - C * z0 - D) / (A * a + B * b + C * c)  # 无需考虑容差报错
        x = x0 + a * t
        y = y0 + b * t
        z = z0 + c * t

        return x, y, z

    # 计算立方体边的连线方程
    line_equations = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            point1 = vertices[i]
            point2 = vertices[j]
            equation = line_equation(point1, point2)
            line_equations.append(equation)

    tolerance = 1e-6
    matching_equations = []
    for equation in line_equations:
        start_point, direction_vector = equation
        # 检查连线方程的方向向量是否与立方体边的方向向量相同或相反
        for edge_direction in lattice_vectors:
            if np.linalg.norm(direction_vector - edge_direction) <= tolerance or np.linalg.norm(
                    direction_vector + edge_direction) <= tolerance:
                matching_equations.append(equation)
                break

    # 计算交点
    intersection_points = []
    for equation in matching_equations:
        start_point, direction_vector = equation
        intersection_point = line_plane_intersection(start_point, direction_vector, plane)
        intersection_points.append(intersection_point)

    return intersection_points

def plot_structure_with_cube(poscar_file, sorted_points, atom):
    """
    使用POSCAR文件的数据在3D坐标轴上绘制一个立方体及其上的点.

    参数:
        poscar_file (str): POSCAR文件的路径.
        sorted_points (list): 要绘制在立方体上的点的列表.
    """

    def plot_cube(ax, vertices):
        edges = [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]]
        ]

        for edge in edges:
            ax.plot3D(*zip(*edge), color='black')

    # 使用pymatgen从POSCAR文件读取你的结构
    file = poscar_file
    structure = Poscar.from_file(file).structure

    # 提取晶胞的顶点
    lattice = structure.lattice.matrix
    vertices = [
        [0, 0, 0],
        lattice[0],
        lattice[0] + lattice[1],
        lattice[1],
        lattice[2],
        lattice[0] + lattice[2],
        lattice[0] + lattice[1] + lattice[2],
        lattice[1] + lattice[2]
    ]

    # 创建一个三维图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制立方体
    plot_cube(ax, vertices)

    x_values = [point[0] for point in sorted_points]
    y_values = [point[1] for point in sorted_points]
    z_values = [point[2] for point in sorted_points]

    ax.scatter(x_values, y_values, z_values, c='blue', marker='o')

    x_values_1 = [point[0] for point in atom]
    y_values_1 = [point[1] for point in atom]
    z_values_1 = [point[2] for point in atom]

    ax.scatter(x_values_1, y_values_1, z_values_1, c='red', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置图形标题
    plt.title('Crystal Structure Cube')

    # 展示图形
    plt.show()

def get_periodic_edge_endpoints(structure, edge):
    """把 StructureGraph 中的一条周期边恢复为真实的两端点笛卡尔坐标。"""
    start_point = structure[edge['from_index']].coords
    end_frac = structure[edge['to_index']].frac_coords + np.array(edge['to_jimage'])
    end_point = structure.lattice.get_cartesian_coords(end_frac)
    return start_point, end_point


def compute_effective_volume_from_crossing_bonds(structure, crossing_bonds, plane, area):
    """
    保留给对比研究用的 V_eff 计算函数，当前主流程不再使用。

    用“被切断键的端点”定义有效厚度和有效体积。

    对每条跨平面的键，局域厚度定义为：
        |d_start| + |d_end|
    其中 d_start 和 d_end 是两个端点到切平面的垂直距离。

    最终:
        t_eff = 所有断键局域厚度的平均值
        V_eff = A * t_eff

    当前脚本的主输出已经切回 V_old；这个函数只在需要比较两种体积定义时保留。
    """
    if not crossing_bonds or area == 0:
        return 0.0, 0.0

    bond_spans = []
    for edge in crossing_bonds:
        start_point, end_point = get_periodic_edge_endpoints(structure, edge)
        start_distance = abs(point_to_plane_distance(start_point, plane))
        end_distance = abs(point_to_plane_distance(end_point, plane))
        bond_spans.append(start_distance + end_distance)

    effective_thickness = float(np.mean(bond_spans))
    effective_volume = effective_thickness * area
    return effective_volume, effective_thickness

def is_inside_parallelepiped_barycentric(point, lattice_vectors, tolerance=1e-5):
    """
    Use barycentric coordinates to determine if a point is inside a parallelepiped.

    Args:
        point (np.array): The point to check.
        lattice_vectors (np.array): 3x3 matrix representing the lattice vectors.
        tolerance (float): Considered floating-point tolerance.

    Returns:
        bool: True if the point is inside the parallelepiped or on its boundary, False otherwise.
    """

    # Solve for the barycentric coordinates
    barycentric_coords = np.linalg.solve(lattice_vectors.T, point)

    # Check if all the barycentric coordinates are within the [0, 1] range (considering tolerance)
    return all(0 - tolerance <= coord <= 1 + tolerance for coord in barycentric_coords)

def build_structure_graph_bonds(poscar_path):
    """
    为 slab 建立周期键图。

    这里保留每条边的 to_jimage，这样后面判断“这条键是否跨过切平面”时，
    不会把跨晶胞的键错误地当成胞内键。
    """
    structure = Structure.from_file(poscar_path)
    bond_structure, oxidation_guessed = prepare_structure_for_bonding(structure)
    cnn = CrystalNN(cation_anion=oxidation_guessed)
    structure_graph = StructureGraph.with_local_env_strategy(bond_structure, cnn)

    unique_edges = []
    seen_edges = set()
    for from_index, to_index, edge_data in structure_graph.graph.edges(data=True):
        to_jimage = tuple(int(value) for value in edge_data.get('to_jimage', (0, 0, 0)))
        edge_id = canonicalize_periodic_edge(from_index, to_index, to_jimage)
        if edge_id in seen_edges:
            continue
        seen_edges.add(edge_id)
        unique_edges.append({
            'from_index': edge_id[0],
            'to_index': edge_id[1],
            'to_jimage': edge_id[2],
        })

    return structure, unique_edges, oxidation_guessed


def canonicalize_periodic_edge(from_index, to_index, to_jimage):
    """给周期边做唯一化，避免同一条物理键被重复统计。"""
    to_jimage = tuple(int(value) for value in to_jimage)
    if from_index > to_index or (from_index == to_index and to_jimage < (0, 0, 0)):
        return to_index, from_index, tuple(-value for value in to_jimage)
    return from_index, to_index, to_jimage


def format_periodic_bond(edge):
    site_a = edge['from_index'] + 1
    site_b = edge['to_index'] + 1
    jimage = edge['to_jimage']
    return f'({site_a}, {site_b})@{jimage}'


def bond_crosses_plane(structure, edge, plane, tolerance=PLANE_TOLERANCE):
    """若一条周期边两端点分居平面两侧，则认为这条键被当前切面切断。"""
    start_point, end_point = get_periodic_edge_endpoints(structure, edge)

    start_distance = point_to_plane_distance(start_point, plane)
    end_distance = point_to_plane_distance(end_point, plane)

    if abs(start_distance) <= tolerance and abs(end_distance) <= tolerance:
        return False
    if abs(start_distance) <= tolerance:
        return abs(end_distance) > tolerance
    if abs(end_distance) <= tolerance:
        return abs(start_distance) > tolerance
    return start_distance * end_distance < 0


def get_crossing_bonds_from_structure_graph(structure, periodic_bonds, plane, tolerance=PLANE_TOLERANCE):
    """收集所有被当前切面切断的键。"""
    crossing_bonds = []
    for edge in periodic_bonds:
        if bond_crosses_plane(structure, edge, plane, tolerance=tolerance):
            crossing_bonds.append(edge)
    return crossing_bonds

def structure_classification(h, k, l, identifier, chemical_formula):
    """
    Organize output files for a given Miller index (h,k,l) by moving them into a designated folder.

    Creates or uses a folder named 'mp-{identifier}-{chemical_formula}/face_{h}_{k}_{l}' relative to the current directory.
    Moves relevant output files (POSCAR files and slab files) into that folder. If the folder already exists, it assumes
    the files have been organized previously and instead removes the newly generated files in the current directory
    to avoid duplicates.

    The material identifier and formula are passed explicitly so the function can be reused from a structured entrypoint.

    Parameters:
        h (int): Miller index h.
        k (int): Miller index k.
        l (int): Miller index l.

    Returns:
        None
    """
    current_path = Path.cwd()
    folder_name = f"mp-{identifier}-{chemical_formula}"
    target_folder = current_path / folder_name / f"face_{h}_{k}_{l}"
    if not target_folder.exists():
        # Create the target directory for this Miller index face
        target_folder.mkdir(parents=True, exist_ok=True)
        # Move specific POSCAR files into the new folder
        for fname in [f"POSCAR_{h}_{k}_{l}.vasp"]:
            src = current_path / fname
            if src.exists():
                shutil.move(str(src), str(target_folder / fname))
        # Move all files starting with "slab_" into the new folder
        for src in current_path.glob("slab_*"):
            if src.is_file():
                shutil.move(str(src), str(target_folder / src.name))
    else:
        # If the folder already exists, remove the newly generated files from the current directory
        for fname in [f"POSCAR_{h}_{k}_{l}.vasp"]:
            file_path = current_path / fname
            if file_path.exists():
                file_path.unlink()
        for src in current_path.glob("slab_*"):
            if src.is_file():
                src.unlink()


def generate_valid_slab(
    structure_path,
    h,
    k,
    l,
    use_d,
    vacuum_size=DEFAULT_MIN_VACUUM_SIZE,
):
    """
    按稳定版逻辑生成 slab。

    与 version_final.py 保持一致的点：
    1. slab 厚度使用 4 * ceil(d)；
    2. center_slab=False；
    3. 只检查并使用 slab_1，不再改成“中间重复周期”策略。

    这样可以把 slab 的层数、原子数和表面起点尽量对齐到稳定版，便于排查
    “结果差异是否来自 slab 生成”。
    """
    miller_index = [int(h), int(k), int(l)]
    structure = Structure.from_file(structure_path)
    slabgen = SlabGenerator(
        structure,
        miller_index,
        center_slab=False,
        min_slab_size=float(use_d),
        min_vacuum_size=vacuum_size,
    )
    all_slabs = slabgen.get_slabs()
    selected_orthogonal_slab = None

    for idx, slab in enumerate(all_slabs):
        orthogonal_slab = slab.get_orthogonal_c_slab()
        if idx == 0:
            selected_orthogonal_slab = orthogonal_slab.copy()
        filename = f"slab_{idx + 1}.vasp"
        Poscar(orthogonal_slab).write_file(filename)

    if not all_slabs:
        return None

    slab_name = 'slab_1.vasp'
    atoms = read(slab_name)
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    zmin = float(np.min(positions[:, 2]))
    zmax = float(np.max(positions[:, 2]))
    slab_thickness = float(abs(zmax - zmin))
    vacuum_thickness = float(cell[2, 2] - slab_thickness)
    if vacuum_thickness >= 4.0:
        final_name = f'POSCAR_{h}_{k}_{l}.vasp'
        if os.path.exists(final_name):
            os.remove(final_name)
        os.rename(slab_name, final_name)
        return {
            'slab_file_name': final_name,
            'target_slab_thickness': float(use_d),
            'actual_slab_thickness': slab_thickness,
            'vacuum_thickness': vacuum_thickness,
            'slab_structure': selected_orthogonal_slab.copy() if selected_orthogonal_slab is not None else None,
        }

    return None


def build_slab_structure_bucket_key(slab_structure):
    """
    为 slab 建一个便宜且稳定的初筛指纹。

    只有当原子数、组分、晶格参数都相近时，才会进入更贵的 StructureMatcher 精确比较。
    """
    species_counts = {}
    for site in slab_structure:
        species_label = site.species_string
        species_counts[species_label] = species_counts.get(species_label, 0) + 1

    return (
        len(slab_structure),
        tuple(round(float(value), 6) for value in slab_structure.lattice.abc),
        tuple(round(float(value), 4) for value in slab_structure.lattice.angles),
        tuple(sorted(species_counts.items())),
    )


def find_matching_slab_representative(slab_structure, slab_registry):
    """
    在已扫描的 slab 代表中查找是否存在“结构完全相同”的代表。

    判断流程：
    1. 先用原子数、组分和晶格参数做粗筛；
    2. 再用 StructureMatcher 做严格结构匹配。
    """
    bucket_key = build_slab_structure_bucket_key(slab_structure)
    for representative in slab_registry.get(bucket_key, []):
        if SLAB_STRUCTURE_MATCHER.fit(representative['structure'], slab_structure):
            return representative, bucket_key
    return None, bucket_key


def cleanup_temp_files(h, k, l):
    file_candidates = [f'POSCAR_{h}_{k}_{l}.vasp']
    for file_name in file_candidates:
        if os.path.exists(file_name):
            os.remove(file_name)

    for file_name in os.listdir(os.getcwd()):
        if file_name.startswith('slab_') and os.path.isfile(file_name):
            os.remove(file_name)


def initialize_method_result(method_name, selection_metric):
    """初始化单个晶面的最优扫描结果记录。"""
    return {
        'method': method_name,
        'selection_metric': selection_metric,
        'best_score': float('inf'),
        'best_area_density': None,
        'best_volume_density': None,
        'mean_volume': None,
        'mean_distances': None,
        'area': None,
        'bond_pairs': [],
        'crossing_bond_count': 0,
        'plane_index': None,
        'plane_height': None,
        'zero_crossing_found': False,
        'oxidation_guessed': None,
        'repeat_period': None,
        'slab_thickness': None,
        'scan_window_width': None,
        'scan_window_start': None,
        'scan_window_end': None,
        'scan_mode': None,
        'scan_plane_count': 0,
        'scan_event_count': 0,
    }


def write_graph_result_to_dataframe(df, index, graph_result, slab_duplicate_of=''):
    """
    把单个 hkl 的扫描结果统一写回 DataFrame。

    若该 hkl 的 slab 与前面某个代表面完全一致，则 `slab_duplicate_of` 记录代表面的 hkl，
    并直接复用代表面的扫描结果，不再重复扫描。
    """
    if graph_result['bond_pairs']:
        df.at[index, 'mean_distances'] = graph_result['mean_distances']
        df.at[index, 'area'] = graph_result['area']
        df.at[index, 'mean_volume'] = graph_result['mean_volume']
        df.at[index, 'bond_pairs'] = ', '.join(graph_result['bond_pairs'])
        df.at[index, 'bond_count'] = graph_result['crossing_bond_count']
        df.at[index, 'bonding_density'] = graph_result['best_volume_density']
        df.at[index, 'bonding_density_area'] = graph_result['best_area_density']
        df.at[index, 'bonding_density_volume'] = graph_result['best_volume_density']
    else:
        df.at[index, 'mean_distances'] = None
        df.at[index, 'area'] = None
        df.at[index, 'mean_volume'] = None
        df.at[index, 'bond_pairs'] = 'unable'
        df.at[index, 'bond_count'] = None
        df.at[index, 'bonding_density'] = None
        df.at[index, 'bonding_density_area'] = None
        df.at[index, 'bonding_density_volume'] = None

    df.at[index, 'bond_method'] = 'structure_graph_jimage'
    df.at[index, 'oxidation_guessed'] = graph_result['oxidation_guessed']
    df.at[index, 'repeat_period'] = graph_result['repeat_period']
    df.at[index, 'slab_thickness'] = graph_result['slab_thickness']
    df.at[index, 'scan_window_width'] = graph_result['scan_window_width']
    df.at[index, 'scan_window_start'] = graph_result['scan_window_start']
    df.at[index, 'scan_window_end'] = graph_result['scan_window_end']
    df.at[index, 'scan_mode'] = graph_result['scan_mode']
    df.at[index, 'scan_plane_count'] = graph_result['scan_plane_count']
    df.at[index, 'scan_event_count'] = graph_result['scan_event_count']
    df.at[index, 'slab_duplicate_of'] = slab_duplicate_of
    df.at[index, 'slab_deduplicated'] = bool(slab_duplicate_of)


def update_method_result(method_result, score, area_density, volume_density, mean_volume,
                         mean_distances, area, bond_pairs, plane_index, plane_height):
    """若当前平面优于历史最优结果，则更新记录。"""
    if score < method_result['best_score']:
        method_result['best_score'] = score
        method_result['best_area_density'] = area_density
        method_result['best_volume_density'] = volume_density
        method_result['mean_volume'] = mean_volume
        method_result['mean_distances'] = mean_distances
        method_result['area'] = area
        method_result['bond_pairs'] = bond_pairs
        method_result['crossing_bond_count'] = len(bond_pairs)
        method_result['plane_index'] = plane_index
        method_result['plane_height'] = plane_height


def build_fixed_scan_heights(zmin, init_d, step_size):
    """按旧版固定步长生成扫描高度。"""
    iterations_begin = int(init_d / step_size) + 1
    iterations_end = int(init_d * 3 / step_size)
    return [float(zmin + step_size * plane_index) for plane_index in range(iterations_begin, iterations_end)]


def merge_close_event_heights(event_heights, tolerance=PLANE_TOLERANCE):
    """对非常接近的事件高度做去重，避免浮点噪声产生无意义的小区间。"""
    merged_heights = []
    for height in sorted(float(value) for value in event_heights):
        if not merged_heights or abs(height - merged_heights[-1]) > tolerance:
            merged_heights.append(height)
    return merged_heights


def collect_scan_event_heights(distance_points, structure, periodic_bonds, z_start, z_end):
    """
    收集会导致目标函数变化的关键高度。

    事件来源：
    1. distance_points 的 z 坐标：决定 V_old 最近上下两层原子的组合何时切换；
    2. 周期键两个端点的 z 坐标：决定断键集合何时发生变化。
    """
    event_heights = []

    for point in distance_points:
        z_value = float(point[2])
        if z_start < z_value < z_end:
            event_heights.append(z_value)

    for edge in periodic_bonds:
        start_point, end_point = get_periodic_edge_endpoints(structure, edge)
        for point in (start_point, end_point):
            z_value = float(point[2])
            if z_start < z_value < z_end:
                event_heights.append(z_value)

    return merge_close_event_heights(event_heights)


def build_event_scan_heights(event_heights, z_start, z_end, tolerance=PLANE_TOLERANCE):
    """
    基于事件高度生成真正需要计算的扫描平面。

    在任意两个相邻事件之间：
    - 断键集合不变；
    - V_old 的最近上下两层原子组合不变；
    因此只需要在区间中点取一个平面即可。
    """
    boundaries = [float(z_start)]
    boundaries.extend(height for height in event_heights if z_start < height < z_end)
    boundaries.append(float(z_end))
    boundaries = merge_close_event_heights(boundaries, tolerance=tolerance)

    scan_heights = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left <= tolerance:
            continue
        scan_heights.append(float((left + right) / 2))

    if not scan_heights and z_end - z_start > tolerance:
        scan_heights.append(float((z_start + z_end) / 2))

    return scan_heights, boundaries


def analyze_slab_file(
    slab_file_name,
    h,
    k,
    l,
    init_d,
    step_size=0.4,
    stop_on_zero_crossing=False,
    scan_mode='auto',
):
    """
    对单个 slab 做平面扫描。

    主流程：
    1. 在稳定版扫描窗口 zmin + ceil(d) 到 zmin + 3*ceil(d) 内取样；
       - fixed: 传统固定步长扫描；
       - event: 只在关键事件区间中点扫描；
       - auto: 自动选择扫描点数量更少的策略；
    2. 计算每个切平面与晶胞的截面积 A；
    3. 用 StructureGraph + jimage 统计被切断的键数 N；
    4. 计算稳定版主指标 N / V_old，并额外保留 N / A；
    5. 选取断键体积密度最低的位置作为该晶面的代表结果。
    """
    processor = PoscarProcessor(slab_file_name)
    vertices, lattice_vectors, _ = processor.process_poscar_file()
    structure = Structure.from_file(slab_file_name)
    atoms = read(slab_file_name)
    positions = atoms.get_positions()
    zmin = float(np.min(positions[:, 2]))
    zmax = float(np.max(positions[:, 2]))
    distance_points = build_old_volume_distance_samples(structure)

    graph_structure, periodic_bonds, oxidation_guessed = build_structure_graph_bonds(slab_file_name)
    graph_result = initialize_method_result('graph', 'volume_density')
    graph_result['oxidation_guessed'] = oxidation_guessed
    graph_result['slab_thickness'] = float(zmax - zmin)
    graph_result['scan_window_width'] = float(2 * init_d)
    graph_result['scan_window_start'] = float(zmin + init_d)
    graph_result['scan_window_end'] = float(zmin + 3 * init_d)

    scan_records = []
    z_start = float(zmin + init_d)
    z_end = float(zmin + 3 * init_d)

    if scan_mode == 'fixed':
        scan_heights = build_fixed_scan_heights(zmin, init_d, step_size)
        graph_result['scan_mode'] = 'fixed'
        graph_result['scan_event_count'] = 0
    elif scan_mode == 'event':
        event_heights = collect_scan_event_heights(distance_points, graph_structure, periodic_bonds, z_start, z_end)
        scan_heights, boundaries = build_event_scan_heights(event_heights, z_start, z_end)
        graph_result['scan_mode'] = 'event'
        graph_result['scan_event_count'] = max(0, len(boundaries) - 2)
    elif scan_mode == 'auto':
        fixed_scan_heights = build_fixed_scan_heights(zmin, init_d, step_size)
        event_heights = collect_scan_event_heights(distance_points, graph_structure, periodic_bonds, z_start, z_end)
        event_scan_heights, boundaries = build_event_scan_heights(event_heights, z_start, z_end)
        if len(event_scan_heights) < len(fixed_scan_heights):
            scan_heights = event_scan_heights
            graph_result['scan_mode'] = 'event'
            graph_result['scan_event_count'] = max(0, len(boundaries) - 2)
        else:
            scan_heights = fixed_scan_heights
            graph_result['scan_mode'] = 'fixed'
            graph_result['scan_event_count'] = max(0, len(boundaries) - 2)
    else:
        raise ValueError("scan_mode 只能是 'auto'、'event' 或 'fixed'")

    graph_result['scan_plane_count'] = len(scan_heights)

    for plane_index, plane_height in enumerate(scan_heights):
        normal_vector = np.array([0, 0, 1])
        moved_point = np.array([0, 0, plane_height], dtype=float)
        plane = np.array(calculate_plane_equation(normal_vector, moved_point))

        # 先求几何截面；若当前平面没有形成有效截面多边形，则跳过。
        intersection_points = compute_intersection_points(vertices, lattice_vectors, plane)
        inside_points = [p for p in intersection_points if is_inside_parallelepiped_barycentric(p, lattice_vectors)]
        sorted_points = filter_and_sort_points(inside_points, normal_vector)
        if not sorted_points:
            continue

        area = compute_3d_polygon_area(sorted_points)
        # 断键统计完全依赖周期键图，不再使用镜像原子或人工扩胞映射。
        crossing_bonds = get_crossing_bonds_from_structure_graph(graph_structure, periodic_bonds, plane)
        # 体积辅助量改回 3.2 风格的 V_old，用于近似描述夹层厚度。
        mean_volume, mean_distances = compute_old_volume_from_plane(distance_points, plane, area)

        record = {
            'plane_index': plane_index,
            'plane_height': float(plane_height),
            'area': area,
            'mean_distances': mean_distances,
            'mean_volume': mean_volume,
            'scan_window_width': float(2 * init_d),
        }

        bond_pairs = [format_periodic_bond(edge) for edge in crossing_bonds]
        crossing_count = len(bond_pairs)
        area_density = crossing_count / area if area else float('inf')
        volume_density = crossing_count / mean_volume if mean_volume else float('inf')

        record['graph_crossing_bond_count'] = crossing_count
        record['graph_area_density'] = area_density
        record['graph_volume_density'] = volume_density
        record['graph_bond_pairs'] = '; '.join(bond_pairs)

        if crossing_count == 0:
            graph_result['zero_crossing_found'] = True
            if stop_on_zero_crossing:
                scan_records.append(record)
                return {
                    'scan_records': scan_records,
                    'graph_result': graph_result,
                    'zero_crossing_found': True,
                }

        if crossing_count != 0:
            update_method_result(
                graph_result,
                # 为了和稳定版保持一致，这里主排序指标改回 N / V_old。
                volume_density,
                area_density,
                volume_density,
                mean_volume,
                mean_distances,
                area,
                bond_pairs,
                plane_index,
                float(plane_height),
            )

        scan_records.append(record)

    return {
        'scan_records': scan_records,
        'graph_result': graph_result,
        'zero_crossing_found': False,
    }


def build_xrd_dataframe(structure_path, two_theta_range=(0, 90), min_intensity=0.1):
    """
    从结构文件自动生成 XRD 反射表。

    这里仍然先用 XRDCalculator 找允许衍射的反射峰，但不再把一个衍射家族
    只压成单个代表面；而是把该代表面的所有点群等价 hkl 展开写入表格。

    这样做的原因是：在当前 slab 生成实现里，像 102 和 012 这样的方向虽然在体相
    对称性上等价，但实际生成出来的 slab 可能不同，因此不能在进入 slab 步骤前就提前合并。

    同时增加一条额外规则：
    - 若 hkl 只是公因数不同，如 001 和 002，则视为同一种晶面，只保留一个代表。
    """
    structure = Structure.from_file(structure_path)
    calculator = XRDCalculator()
    pattern = calculator.get_pattern(structure, scaled=True, two_theta_range=two_theta_range)
    point_group_ops = SpacegroupAnalyzer(structure, symprec=1e-3).get_point_group_operations(cartesian=False)

    rows = []
    seen_plane_directions = set()
    for peak_index, two_theta in enumerate(pattern.x):
        intensity = float(pattern.y[peak_index])
        if intensity < min_intensity:
            continue

        d_spacing = float(pattern.d_hkls[peak_index])
        for hkl_info in pattern.hkls[peak_index]:
            representative_hkl = convert_diffraction_hkl_to_three_index(hkl_info['hkl'])
            if representative_hkl == (0, 0, 0):
                continue

            equivalent_hkls = expand_equivalent_hkls(representative_hkl, point_group_ops)
            for hkl in equivalent_hkls:
                reduced_hkl = canonicalize_plane_direction(hkl)
                if reduced_hkl in seen_plane_directions:
                    continue
                seen_plane_directions.add(reduced_hkl)
                rows.append({
                    'h': hkl[0],
                    'k': hkl[1],
                    'l': hkl[2],
                    'reduced_h': reduced_hkl[0],
                    'reduced_k': reduced_hkl[1],
                    'reduced_l': reduced_hkl[2],
                    'd(Å)': d_spacing,
                    '2theta': float(two_theta),
                    'intensity': intensity,
                    'multiplicity': int(hkl_info.get('multiplicity', 1)),
                    'family_h': representative_hkl[0],
                    'family_k': representative_hkl[1],
                    'family_l': representative_hkl[2],
                    'equivalent_count': len(equivalent_hkls),
                })

    xrd_df = pd.DataFrame(rows)
    if xrd_df.empty:
        return xrd_df

    xrd_df = xrd_df.sort_values(
        by=['d(Å)', 'intensity', 'h', 'k', 'l'],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return xrd_df


def write_xrd_table(structure_path, output_dir, two_theta_range=(0, 90), min_intensity=0.1):
    """为单个结构写出 XRD Excel 表。"""
    structure_path = Path(structure_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xrd_df = build_xrd_dataframe(
        structure_path,
        two_theta_range=two_theta_range,
        min_intensity=min_intensity,
    )
    output_path = output_dir / f'{structure_path.stem}-XRD.xlsx'
    xrd_df.to_excel(output_path, index=False)
    return output_path, len(xrd_df)


def generate_xrd_tables_for_directory(input_dir, output_dir, two_theta_range=(0, 90), min_intensity=0.1):
    """递归扫描目录中的 .vasp 文件，并批量生成 XRD Excel。"""
    input_dir = Path(input_dir)
    vasp_files = sorted(input_dir.rglob('*.vasp'))
    generated_files = []

    for structure_path in vasp_files:
        output_path, row_count = write_xrd_table(
            structure_path,
            output_dir,
            two_theta_range=two_theta_range,
            min_intensity=min_intensity,
        )
        generated_files.append((output_path, row_count))

    return generated_files


def ensure_xrd_table(poscar_file, poscar_folder, excel_folder, two_theta_range=(0, 90), min_intensity=0.1):
    """当 batch 缺少对应 XRD 表时，按需即时生成。"""
    structure_path = Path(poscar_folder) / poscar_file
    output_path, _ = write_xrd_table(
        structure_path,
        excel_folder,
        two_theta_range=two_theta_range,
        min_intensity=min_intensity,
    )
    return output_path


def filter_xrd_rows_by_intensity(df, min_intensity):
    """
    按 XRD 相对强度阈值过滤待扫描的 hkl。

    这里的 intensity 来自 XRDCalculator(scaled=True)，已经按最大峰归一化。
    如果 min_intensity <= 0，则视为不过滤，保留全部 hkl。
    """
    if min_intensity is None or min_intensity <= 0:
        return df

    if 'intensity' not in df.columns:
        print('Warning: intensity column is missing in the XRD table, skip intensity filtering.')
        return df

    filtered_df = df[df['intensity'] >= float(min_intensity)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def deduplicate_hkls_by_common_factor(df):
    """
    按“去公因数后的晶面方向”去重。

    例如 001 和 002 只保留一个代表；但 012 和 102 仍然会同时保留。
    这里保留 DataFrame 当前顺序中的第一项，因此如果 XRD 表已按 d 从大到小排序，
    通常会优先保留低阶反射。
    """
    required_columns = {'h', 'k', 'l'}
    if not required_columns.issubset(df.columns):
        return df

    kept_rows = []
    seen_plane_directions = set()
    for _, row in df.iterrows():
        hkl = (int(row['h']), int(row['k']), int(row['l']))
        reduced_hkl = canonicalize_plane_direction(hkl)
        if reduced_hkl in seen_plane_directions:
            continue

        row_dict = row.to_dict()
        row_dict['reduced_h'] = reduced_hkl[0]
        row_dict['reduced_k'] = reduced_hkl[1]
        row_dict['reduced_l'] = reduced_hkl[2]
        kept_rows.append(row_dict)
        seen_plane_directions.add(reduced_hkl)

    deduplicated_df = pd.DataFrame(kept_rows)
    deduplicated_df.reset_index(drop=True, inplace=True)
    return deduplicated_df


def parse_material_identity(poscar_name):
    """从 mp-xxx-Formula.vasp 文件名中解析材料编号和化学式。"""
    parts = poscar_name.split('-')
    if len(parts) < 3:
        raise ValueError(f'Unexpected POSCAR filename: {poscar_name}')
    identifier = parts[1]
    chemical_formula = parts[2].split('.')[0]
    return identifier, chemical_formula


def move_worker_outputs_to_root(workdir, output_root, folder_name, result_file_name):
    """
    将并行 worker 目录中的结果搬回主输出目录。

    说明：
    1. worker 在独立工作目录中运行，避免 slab_*.vasp 等临时文件互相覆盖；
    2. 任务结束后，再把 after_deal 和 face_* 结果移回主目录。
    """
    workdir = Path(workdir)
    output_root = Path(output_root)

    local_after_deal = workdir / 'after_deal' / result_file_name
    target_after_deal = output_root / 'after_deal'
    target_after_deal.mkdir(parents=True, exist_ok=True)
    if local_after_deal.exists():
        target_after_file = target_after_deal / result_file_name
        if target_after_file.exists():
            target_after_file.unlink()
        shutil.move(str(local_after_deal), str(target_after_file))

    local_material_folder = workdir / folder_name
    if local_material_folder.exists():
        target_material_folder = output_root / folder_name
        if target_material_folder.exists():
            shutil.rmtree(target_material_folder)
        shutil.move(str(local_material_folder), str(target_material_folder))

    shutil.rmtree(workdir, ignore_errors=True)


def process_single_material_task(task):
    """
    处理单个材料。

    并行策略说明：
    - 当前只在“不同材料”这一层并行；
    - 单个材料内部的 hkl 扫描仍保持串行，以避免改变原有 zero-crossing 逻辑；
    - 每个 worker 使用独立工作目录，避免 slab_*.vasp / POSCAR_*.vasp 临时文件冲突。
    """
    poscar_path = Path(task['poscar_path']).resolve()
    excel_folder = Path(task['excel_folder']).resolve()
    output_root = Path(task['output_root']).resolve()
    step_size = float(task['step_size'])
    scan_mode = str(task['scan_mode'])
    auto_generate_xrd = bool(task['auto_generate_xrd'])
    xrd_two_theta_min = float(task['xrd_two_theta_min'])
    xrd_two_theta_max = float(task['xrd_two_theta_max'])
    xrd_min_intensity = float(task['xrd_min_intensity'])
    deduplicate_identical_slabs = bool(task['deduplicate_identical_slabs'])

    poscar_name = poscar_path.name
    identifier, chemical_formula = parse_material_identity(poscar_name)
    folder_name = f'mp-{identifier}-{chemical_formula}'
    result_file_name = f'mp-{identifier}-{chemical_formula}.xlsx'

    work_root = Path(task['work_root']).resolve()
    workdir = work_root / folder_name
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    messages = []
    try:
        os.chdir(workdir)
        Path('after_deal').mkdir(exist_ok=True)

        excel_file_name = f'mp-{identifier}-{chemical_formula}-XRD.xlsx'
        excel_file_path = excel_folder / excel_file_name

        if not excel_file_path.exists() and auto_generate_xrd:
            ensure_xrd_table(
                poscar_name,
                str(poscar_path.parent),
                str(excel_folder),
                two_theta_range=(xrd_two_theta_min, xrd_two_theta_max),
                min_intensity=xrd_min_intensity,
            )

        if not excel_file_path.exists():
            return {
                'status': 'missing_xrd',
                'material_name': folder_name,
                'messages': messages,
            }

        df = pd.read_excel(excel_file_path, engine='openpyxl')
        original_hkl_count = len(df)
        df = filter_xrd_rows_by_intensity(df, xrd_min_intensity)
        intensity_filtered_hkl_count = len(df)
        df = deduplicate_hkls_by_common_factor(df)
        filtered_hkl_count = len(df)

        if filtered_hkl_count == 0:
            messages.append(
                f'{poscar_name}: no hkl remains after intensity/common-factor filtering.'
            )
            return {
                'status': 'no_hkl',
                'material_name': folder_name,
                'messages': messages,
            }

        if intensity_filtered_hkl_count != original_hkl_count:
            messages.append(
                f'{poscar_name}: using {intensity_filtered_hkl_count}/{original_hkl_count} hkls '
                f'with intensity >= {xrd_min_intensity}.'
            )
        if filtered_hkl_count != intensity_filtered_hkl_count:
            messages.append(
                f'{poscar_name}: merged common-factor-equivalent hkls, '
                f'keeping {filtered_hkl_count}/{intensity_filtered_hkl_count}.'
            )

        os.mkdir(folder_name)
        break_value = 0
        full_poscar_path = str(poscar_path)
        slab_registry = {}
        valid_slab_count = 0
        duplicate_slab_count = 0
        unique_slab_scan_count = 0

        for index, row in df.iterrows():
            if break_value == 1:
                old_file = full_poscar_path
                new_file_name = f'vdW-{h}-{k}-{l}-{poscar_name}'
                new_file_path = os.path.join(str(poscar_path.parent), new_file_name)
                os.rename(old_file, new_file_path)
                break

            h = int(row['h'])
            k = int(row['k'])
            l = int(row['l'])
            d = float(row['d(Å)'])
            init_d = int(np.ceil(d))
            use_d = int(init_d * 4)

            slab_context = generate_valid_slab(full_poscar_path, h, k, l, use_d)
            if slab_context is None:
                continue
            valid_slab_count += 1

            slab_structure = slab_context.get('slab_structure')
            matching_representative = None
            slab_bucket_key = None
            if deduplicate_identical_slabs and slab_structure is not None:
                matching_representative, slab_bucket_key = find_matching_slab_representative(slab_structure, slab_registry)
                if matching_representative is not None:
                    duplicate_slab_count += 1
                    write_graph_result_to_dataframe(
                        df,
                        index,
                        matching_representative['graph_result'],
                        slab_duplicate_of=format_hkl_label(matching_representative['hkl']),
                    )
                    cleanup_temp_files(h, k, l)
                    df.to_excel(os.path.join('after_deal', result_file_name), index=False)
                    continue

            result = analyze_slab_file(
                slab_context['slab_file_name'],
                h,
                k,
                l,
                init_d,
                step_size=step_size,
                stop_on_zero_crossing=True,
                scan_mode=scan_mode,
            )

            if result['zero_crossing_found']:
                break_value = 1
                cleanup_temp_files(h, k, l)
                continue

            graph_result = result['graph_result']
            write_graph_result_to_dataframe(df, index, graph_result)

            if deduplicate_identical_slabs and slab_structure is not None:
                registry_entries = slab_registry.setdefault(slab_bucket_key, [])
                registry_entries.append({
                    'structure': slab_structure.copy(),
                    'hkl': (h, k, l),
                    'graph_result': graph_result,
                })
                unique_slab_scan_count += 1

            if graph_result['bond_pairs']:
                structure_classification(h, k, l, identifier, chemical_formula)
                df.to_excel(os.path.join('after_deal', result_file_name), index=False)
            else:
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                    for file_name in os.listdir(os.getcwd()):
                        if file_name.startswith('slab_'):
                            shutil.move(file_name, os.path.join(folder_name, file_name))
                else:
                    for file_name in os.listdir(os.getcwd()):
                        if file_name.startswith('slab_'):
                            os.remove(file_name)
                df.to_excel(os.path.join('after_deal', result_file_name), index=False)

        if deduplicate_identical_slabs and duplicate_slab_count > 0:
            messages.append(
                f'{poscar_name}: compared {valid_slab_count} valid slabs and reused '
                f'{duplicate_slab_count} duplicate slabs; actually scanned {unique_slab_scan_count} unique slabs.'
            )

        final_result_path = workdir / 'after_deal' / result_file_name
        if not final_result_path.exists():
            df.to_excel(final_result_path, index=False)

        move_worker_outputs_to_root(workdir, output_root, folder_name, result_file_name)
        return {
            'status': 'ok',
            'material_name': folder_name,
            'messages': messages,
        }
    except Exception:
        return {
            'status': 'error',
            'material_name': folder_name,
            'messages': messages,
            'traceback': traceback.format_exc(),
        }
    finally:
        os.chdir(original_cwd)


def run_generate_xrd_mode(args):
    """命令行入口：只生成 XRD 表，不做剥离分析。"""
    generated_files = generate_xrd_tables_for_directory(
        args.input_dir,
        args.output_dir,
        two_theta_range=(args.two_theta_min, args.two_theta_max),
        min_intensity=args.min_intensity,
    )

    print(f'generated_count={len(generated_files)}')
    for output_path, row_count in generated_files[:20]:
        print(f'{output_path},{row_count}')


def run_batch_mode(args):
    """
    命令行入口：批量处理 mp-*.vasp。

    对每个材料：
    1. 若没有 XRD 表，则自动生成；
    2. 逐个 hkl 生成 slab；
    3. 扫描切面并统计断键；
    4. 把最优结果写回 Excel，并整理输出文件。
    """
    poscar_folder = args.poscar_folder
    excel_folder = args.excel_folder
    requested_poscar_name = args.poscar_name
    output_root = Path.cwd()
    Path('after_deal').mkdir(exist_ok=True)
    Path(excel_folder).mkdir(parents=True, exist_ok=True)
    work_root = output_root / '.parallel_tmp'
    work_root.mkdir(parents=True, exist_ok=True)
    poscar_files = os.listdir(poscar_folder)
    excel_files = set(os.listdir(excel_folder))

    filtered_poscar_files = [f for f in poscar_files if f.startswith('mp-') and f.endswith('.vasp')]
    if requested_poscar_name:
        filtered_poscar_files = [f for f in filtered_poscar_files if f == requested_poscar_name]
    tasks = []
    for poscar_file in filtered_poscar_files:
        identifier, chemical_formula = parse_material_identity(poscar_file)
        folder_name = f'mp-{identifier}-{chemical_formula}'
        if (output_root / folder_name).exists():
            print(f'{folder_name} already exists, skipping')
            continue

        tasks.append({
            'poscar_path': os.path.join(poscar_folder, poscar_file),
            'excel_folder': str(Path(excel_folder).resolve()),
            'output_root': str(output_root.resolve()),
            'work_root': str(work_root.resolve()),
            'step_size': args.step_size,
            'scan_mode': args.scan_mode,
            'auto_generate_xrd': args.auto_generate_xrd,
            'xrd_two_theta_min': args.xrd_two_theta_min,
            'xrd_two_theta_max': args.xrd_two_theta_max,
            'xrd_min_intensity': args.xrd_min_intensity,
            'deduplicate_identical_slabs': args.deduplicate_identical_slabs,
        })

    if not tasks:
        print('No materials to process.')
        return

    print(f'scan_mode={args.scan_mode}')
    print(f'parallel_workers={args.parallel_workers}')
    print(f'deduplicate_identical_slabs={args.deduplicate_identical_slabs}')
    if requested_poscar_name:
        print(f'poscar_name={requested_poscar_name}')
    print(f'materials_to_process={len(tasks)}')

    if args.parallel_workers <= 1:
        for task in tqdm(tasks, desc='Processing files'):
            result = process_single_material_task(task)
            for message in result.get('messages', []):
                print(message)
            if result.get('status') == 'error':
                print(f"[ERROR] {result['material_name']}")
                print(result.get('traceback', ''))
        return

    try:
        with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
            futures = [executor.submit(process_single_material_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing files'):
                result = future.result()
                for message in result.get('messages', []):
                    print(message)
                if result.get('status') == 'error':
                    print(f"[ERROR] {result['material_name']}")
                    print(result.get('traceback', ''))
    except Exception as exc:
        print(f'Parallel execution failed ({exc}), falling back to serial mode.')
        for task in tqdm(tasks, desc='Processing files'):
            result = process_single_material_task(task)
            for message in result.get('messages', []):
                print(message)
            if result.get('status') == 'error':
                print(f"[ERROR] {result['material_name']}")
                print(result.get('traceback', ''))


def build_arg_parser():
    """定义命令行参数。默认入口是 batch。"""
    parser = argparse.ArgumentParser(
        description='晶体剥离面分析脚本。',
        epilog=CONFIG_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command')

    batch_parser = subparsers.add_parser(
        'batch',
        help='完整跑剥离分析。',
        description='读取 mp-*.vasp 和 XRD 表，对每个 hkl 生成 slab、扫描切面并统计断键。',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    batch_parser.add_argument('--poscar-folder', default='.', help='待分析的 mp-*.vasp 所在目录。')
    batch_parser.add_argument('--excel-folder', default='XRD', help='XRD Excel 所在目录。')
    batch_parser.add_argument('--poscar-name', default=None, help='只处理指定的单个 mp-*.vasp 文件名。')
    batch_parser.add_argument('--step-size', type=float, default=0.4, help='切面扫描步长，默认 0.4。')
    batch_parser.add_argument(
        '--scan-mode',
        choices=['auto', 'event', 'fixed'],
        default='auto',
        help='切面扫描策略：auto=自动择优，event=事件点扫描，fixed=固定步长扫描。',
    )
    batch_parser.add_argument(
        '--parallel-workers',
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help='按材料并行的进程数；1 表示串行。',
    )
    batch_parser.add_argument(
        '--deduplicate-identical-slabs',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否在生成 slab 后按结构去重；默认开启。',
    )
    batch_parser.add_argument(
        '--auto-generate-xrd',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否在缺少 XRD.xlsx 时自动生成，默认开启。',
    )
    batch_parser.add_argument('--xrd-two-theta-min', type=float, default=0.0, help='自动生成 XRD 时的最小 2theta。')
    batch_parser.add_argument('--xrd-two-theta-max', type=float, default=90.0, help='自动生成 XRD 时的最大 2theta。')
    batch_parser.add_argument(
        '--xrd-min-intensity',
        type=float,
        default=0.1,
        help='batch 真正扫描前的强度过滤阈值；0 表示保留全部 hkl。',
    )

    xrd_parser = subparsers.add_parser(
        'generate-xrd',
        help='只生成 XRD 表。',
        description='递归扫描目录中的 .vasp 文件，并写出对应的 XRD Excel。',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    xrd_parser.add_argument('--input-dir', required=True, help='输入结构目录，会递归扫描所有 .vasp。')
    xrd_parser.add_argument('--output-dir', default='XRD', help='XRD Excel 输出目录。')
    xrd_parser.add_argument('--two-theta-min', type=float, default=0.0, help='XRD 最小 2theta。')
    xrd_parser.add_argument('--two-theta-max', type=float, default=90.0, help='XRD 最大 2theta。')
    xrd_parser.add_argument(
        '--min-intensity',
        type=float,
        default=0.1,
        help='只把强度大于等于该值的面写入 XRD.xlsx；0 表示不过滤。',
    )

    return parser


def build_args_from_script_config():
    """
    当用户直接运行 `python version_4.0.py` 时，使用脚本顶部配置区生成参数。
    """
    mode = SCRIPT_RUN_CONFIG.get('mode', 'batch')

    if mode == 'generate-xrd':
        config = SCRIPT_RUN_CONFIG.get('generate_xrd', {})
        return argparse.Namespace(
            command='generate-xrd',
            input_dir=config.get('input_dir', '.'),
            output_dir=config.get('output_dir', 'XRD'),
            two_theta_min=float(config.get('two_theta_min', 0.0)),
            two_theta_max=float(config.get('two_theta_max', 90.0)),
            min_intensity=float(config.get('min_intensity', 0.1)),
        )

    if mode == 'batch':
        config = SCRIPT_RUN_CONFIG.get('batch', {})
        return argparse.Namespace(
            command='batch',
            poscar_folder=config.get('poscar_folder', '.'),
            excel_folder=config.get('excel_folder', 'XRD'),
            poscar_name=config.get('poscar_name', None),
            step_size=float(config.get('step_size', 0.4)),
            scan_mode=str(config.get('scan_mode', 'auto')),
            parallel_workers=int(config.get('parallel_workers', DEFAULT_PARALLEL_WORKERS)),
            deduplicate_identical_slabs=bool(config.get('deduplicate_identical_slabs', True)),
            auto_generate_xrd=bool(config.get('auto_generate_xrd', True)),
            xrd_two_theta_min=float(config.get('xrd_two_theta_min', 0.0)),
            xrd_two_theta_max=float(config.get('xrd_two_theta_max', 90.0)),
            xrd_min_intensity=float(config.get('xrd_min_intensity', 0.1)),
        )

    raise ValueError("SCRIPT_RUN_CONFIG['mode'] 只能是 'batch' 或 'generate-xrd'")


def main():
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        args = build_args_from_script_config()
        print(f'Using built-in script config: mode={args.command}')
    else:
        args = parser.parse_args()
        if args.command is None:
            args = parser.parse_args(['batch'])

    if args.command == 'generate-xrd':
        run_generate_xrd_mode(args)
        return

    if args.command in (None, 'batch'):
        run_batch_mode(args)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
