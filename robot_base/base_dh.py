import numpy as np
from typing import List, Optional

# 假设这些类在同级目录的其他文件中已经定义好
from robot_base.datatypes import Velocities, GaitConfig
from robot_base.leg_dh import QuadrupedLeg

class QuadrupedBase:
    def __init__(self, gait_conf: Optional[GaitConfig] = None):
        """
        初始化四足机器人的基座类。
        """
        self.speed_: Velocities = Velocities()
        
        # 实例化四条腿：左前(lf), 右前(rf), 左后(lh), 右后(rh)
        self.lf: QuadrupedLeg = QuadrupedLeg()
        self.rf: QuadrupedLeg = QuadrupedLeg()
        self.lh: QuadrupedLeg = QuadrupedLeg()
        self.rh: QuadrupedLeg = QuadrupedLeg()

        # 将四条腿按固定顺序存入列表，方便循环遍历
        self.legs: List[QuadrupedLeg] = [
            self.lf, 
            self.rf, 
            self.lh, 
            self.rh
        ]

        # 默认初始化一个 GaitConfig
        self.gait_config: GaitConfig = GaitConfig()

        # 如果传入了外部配置，则应用配置
        if gait_conf is not None:
            self.set_gait_config(gait_conf)
        else:
            self.set_gait_config(self.gait_config)

    def _get_knee_direction(self, direction_char: str) -> int:
        """
        解析膝盖朝向字符。对应 C++ 的 getKneeDirection。
        Python 中没有 switch 语句，通常使用 if/elif 或字典映射。
        """
        if direction_char == '>':
            return -1
        elif direction_char == '<':
            return 1
        else:
            return -1

    def get_joint_positions(self) -> List[float]:
        """
        获取当前机器人的所有关节角度。
        返回包含 28 个 float 的列表 [lf_hip, lf_up, lf_low, rf_hip, ...]
        """
        joint_positions = []
        for leg in self.legs:
            joint_positions.append(leg.spine.theta)
            joint_positions.append(leg.hip_1.theta)
            joint_positions.append(leg.hip_2.theta)
            joint_positions.append(leg.upper_leg.theta)
            joint_positions.append(leg.lower_leg.theta)
            joint_positions.append(leg.ankel.theta)
            joint_positions.append(leg.foot.theta)
            
        return joint_positions

    def get_foot_positions(self) -> List[np.ndarray]:
        """
        获取四条腿足端相对于基座(Base)坐标系的 4x4 变换矩阵。
        对应 C++ 中使用 geometry::Transformation 数组的逻辑。
        """
        foot_positions = []
        for leg in self.legs:
            foot_positions.append(leg.foot_from_base())
            
        return foot_positions

    def update_joint_positions(self, joints: List[float]) -> None:
        """
        更新机器人的关节角度。
        输入应该为读取到的实际电机的角度值，长度必须为 28。
        """
        if len(joints) != 28:
            raise ValueError(f"Expected 28 joint angles, got {len(joints)}")

        for i, leg in enumerate(self.legs):
            index = i * 7
            leg.spine.theta = joints[index]
            leg.hip_1.theta = joints[index + 1]
            leg.hip_2.theta = joints[index + 2]
            leg.upper_leg.theta = joints[index + 3]
            leg.lower_leg.theta = joints[index + 4]
            leg.ankel.theta = joints[index + 5]
            leg.foot.theta = joints[index + 6]

    def set_gait_config(self, gait_conf: GaitConfig) -> None:
        """
        设置全局步态配置，并将其同步给每一条腿。
        """
        self.gait_config = gait_conf

        # GaitConfig 的 knee_orientation 通常是一个类似 ">>" 或 "><" 的字符串
        # ">>" 表示前腿和后腿膝盖都向后弯（狗腿结构）
        # "><" 表示前腿向后，后腿向前（蜘蛛结构）
        knee_str = self.gait_config.knee_orientation
        if len(knee_str) < 2:
            knee_str = ">>" # 防止字符串越界的保护措施

        for i, leg in enumerate(self.legs):
            leg.leg_id = i
            
            # 索引 0, 1 是前腿(lf, rf)，提取第 0 个字符
            if i < 2:
                dir_val = self._get_knee_direction(knee_str[0])
            # 索引 2, 3 是后腿(lh, rh)，提取第 1 个字符
            else:
                dir_val = self._get_knee_direction(knee_str[1])
                
            leg.knee_direction = dir_val

            # 将全局配置的引用传给每一条腿
            leg.gait_config = self.gait_config

    def setup_robot_geometry(self, l1: float, l2: float, 
                             l3: float, l4: float, l5: float, l6: float, l7: float) -> None:
        """
        初始化机器人的运动学连杆参数 (单位: 米)。
        使用 set_arguments 给 Joint 类内部的 dh参数 赋值。
        对角关系的两条腿theta取反, 坐标(Y)相同 
        左右镜像的腿theta相同, 坐标(Y)相反 
        前后镜像theta相反, 坐标符号(Y)相反
        """

        # ---------------------------------------------------------
        # 左前腿 (LF) - 位于第一象限 (+, +)
        # ---------------------------------------------------------
        self.lf.spine.set_arguments(0.0, l2, np.pi/2, joint_type="revolute") # 脊柱中心固定不动
        self.lf.hip_1.set_arguments(l1, l3, 0.0) # 髋关节安装位置
        self.lf.hip_2.set_arguments(0.0, l4, -np.pi/2) # 向右外展为负
        self.lf.upper_leg.set_arguments(0.0, l5, 0.0)
        self.lf.lower_leg.set_arguments(0.0, l6, -np.pi/2, joint_type="revolute", offset=-np.pi/2) # 膝盖向后弯曲，默认偏置 -90 度
        self.lf.ankel.set_arguments(0.0, l7, 0.0, joint_type="revolute", offset=np.pi/2) # 踝关节固定不动
        self.lf.foot.set_arguments(0.0, 0.0, 0.0)
        # ---------------------------------------------------------
        # 右前腿 (RF) - 位于第四象限 (+, -)
        # ---------------------------------------------------------
        self.rf.spine.set_arguments(0.0, l2, -np.pi/2, joint_type="revolute") # 脊柱中心固定不动
        self.rf.hip_1.set_arguments(l1, l3, 0.0) # 髋关节安装位置
        self.rf.hip_2.set_arguments(0.0, l4, np.pi/2) # 向右外展为负
        self.rf.upper_leg.set_arguments(0.0, l5, 0.0)
        self.rf.lower_leg.set_arguments(0.0, l6, np.pi/2, joint_type="revolute", offset=np.pi/2) # 膝盖向后弯曲，默认偏置 -90 度
        self.rf.ankel.set_arguments(0.0, l7, 0.0, joint_type="revolute", offset=-np.pi/2) # 踝关节固定不动
        self.rf.foot.set_arguments(0.0, 0.0, 0.0)
        # ---------------------------------------------------------
        # 左后腿 (LH) - 位于第二象限 (-, +)
        # ---------------------------------------------------------
        self.lh.spine.set_arguments(0.0, l2, -np.pi/2, joint_type="revolute", offset=0.0) # 脊柱中心固定不动
        self.lh.hip_1.set_arguments(l1, l3, 0.0) # 髋关节安装位置
        self.lh.hip_2.set_arguments(0.0, l4, np.pi/2) # 向右外展为负
        self.lh.upper_leg.set_arguments(0.0, l5, 0.0)
        self.lh.lower_leg.set_arguments(0.0, l6, np.pi/2, joint_type="revolute", offset=np.pi/2) # 膝盖向后弯曲，默认偏置 -90 度
        self.lh.ankel.set_arguments(0.0, l7, 0.0, joint_type="revolute", offset=-np.pi/2) # 踝关节固定不动
        self.lh.foot.set_arguments(0.0, 0.0, 0.0)
        # ---------------------------------------------------------
        # 右后腿 (RH) - 位于第三象限 (-, -)
        # ---------------------------------------------------------
        self.rh.spine.set_arguments(0.0, l2, np.pi/2, joint_type="revolute") # 脊柱中心固定不动
        self.rh.hip_1.set_arguments(l1, l3, 0.0) # 髋关节安装位置
        self.rh.hip_2.set_arguments(0.0, l4, -np.pi/2) # 向右外展为负
        self.rh.upper_leg.set_arguments(0.0, l5, 0.0)
        self.rh.lower_leg.set_arguments(0.0, l6, -np.pi/2, joint_type="revolute", offset=-np.pi/2) # 膝盖向后弯曲，默认偏置 -90 度
        self.rh.ankel.set_arguments(0.0, l7, 0.0, joint_type="revolute", offset=np.pi/2) # 踝关节固定不动
        self.rh.foot.set_arguments(0.0, 0.0, 0.0)