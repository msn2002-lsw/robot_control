from dataclasses import dataclass, field
# 线速度
@dataclass
class Linear:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

# 角速度
@dataclass
class Angular:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

# 速度类
@dataclass
class Velocities:
    linear: Linear = field(default_factory=Linear)
    angular: Angular = field(default_factory=Angular)

# 四元数表示位姿
@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

# 点
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

# 角度
@dataclass
class Euler:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

# 位姿
@dataclass
class Pose:
    position: Point = field(default_factory=Point)
    orientation: Euler = field(default_factory=Euler)

# 加速度计(IMU)
@dataclass
class Accelerometer:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

# 角速度计(IMU),陀螺仪
@dataclass
class Gyroscope:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

# 磁力计数据(IMU)
@dataclass
class Magnetometer:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

# 步态信息
@dataclass
class GaitConfig:
    knee_orientation: str = ">>"            # 膝盖朝向
    pantograph_leg: bool = False            # 受电弓式腿
    odom_scaler: float = 0.0                # 里程计缩放比例
    max_linear_velocity_x: float = 0.0      # 最大X轴线速度
    max_linear_velocity_y: float = 0.0      # 最大Y轴线速度
    max_angular_velocity_z: float = 0.0     # 最大偏航速度
    com_x_translation: float = 0.0          # 质心X轴偏移
    swing_height: float = 0.0               # 摆动高度
    stance_depth: float = 0.0               # 支撑深度
    stance_duration: float = 0.0            # 支撑时长
    nominal_height: float = 0.0             # 额定高度
    phase_offset: list[float] =field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0]) # 相位偏移