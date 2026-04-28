import numpy as np
# ==========================================
# 齐次变换矩阵辅助函数 (替代 geometry::Transformation 的底层逻辑)
# ==========================================
def translate_mat(x: float, y: float, z: float) -> np.ndarray:
    """生成 4x4 平移矩阵"""
    return np.array([
        [1.0, 0.0, 0.0,  x ],
        [0.0, 1.0, 0.0,  y ],
        [0.0, 0.0, 1.0,  z ],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_x_mat(theta: float) -> np.ndarray:
    """生成绕 X 轴旋转的 4x4 矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,  c,   -s,  0.0],
        [0.0,  s,    c,  0.0],
        [0.0, 0.0,  0.0, 1.0]
    ])

def rotate_y_mat(theta: float) -> np.ndarray:
    """生成绕 Y 轴旋转的 4x4 矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  0.0,  s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_z_mat(theta: float) -> np.ndarray:
    """生成绕 Z 轴旋转的 4x4 矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [  c,  -s, 0.0, 0.0],
        [  s,   c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """辅助函数：将静态欧拉角转换为 4x4 旋转矩阵 (假设顺序为 Z-Y-X 或根据你的系统定义)"""
    # 简化的独立相乘，实际工程中通常封装为一个合并后的欧拉矩阵
    R_x = rotate_x_mat(roll)
    R_y = rotate_y_mat(pitch)
    R_z = rotate_z_mat(yaw)
    return R_z @ R_y @ R_x 