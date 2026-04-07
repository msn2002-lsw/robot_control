from typing import Optional

# 假设前面的数据类存放在同级目录的 quadruped_components 模块中
from robot_base.datatypes import Point, Euler

class Joint:
    def __init__(
        self, 
        pos_x: float = 0.0, 
        pos_y: float = 0.0, 
        pos_z: float = 0.0, 
        or_r: float = 0.0, 
        or_p: float = 0.0, 
        or_y: float = 0.0
    ):
        """
        初始化 Joint 类。
        在 Python 中，我们可以通过默认参数 (0.0) 将 C++ 的无参构造函数
        和带参构造函数合并为一个 __init__ 方法。
        """
        self._theta: float = 0.0
        self._translation: Point = Point(pos_x, pos_y, pos_z)
        self._rotation: Euler = Euler(or_r, or_p, or_y)

    # ==========================================
    # Theta 的 Getter 和 Setter (对应 C++ 的重载)
    # ==========================================
    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, angle: float) -> None:
        self._theta = angle

    # ==========================================
    # 批量设置方法
    # ==========================================
    def set_translation(self, x: float, y: float, z: float) -> None:
        self._translation.x = x
        self._translation.y = y
        self._translation.z = z

    def set_rotation(self, roll: float, pitch: float, yaw: float) -> None:
        self._rotation.roll = roll
        self._rotation.pitch = pitch
        self._rotation.yaw = yaw

    def set_origin(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        self.set_translation(x, y, z)
        self.set_rotation(roll, pitch, yaw)

    # ==========================================
    # 只读属性 (对应 C++ 中的 const getters)
    # ==========================================
    @property
    def x(self) -> float:
        return self._translation.x

    @property
    def y(self) -> float:
        return self._translation.y

    @property
    def z(self) -> float:
        return self._translation.z

    @property
    def roll(self) -> float:
        return self._rotation.roll

    @property
    def pitch(self) -> float:
        return self._rotation.pitch

    @property
    def yaw(self) -> float:
        return self._rotation.yaw