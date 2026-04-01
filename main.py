import time
import math
import threading
import numpy as np
from typing import List, Optional

# ==========================================
# 假设以下为您之前转换好的各个模块
# 根据实际的文件夹结构进行 import
# ==========================================
from robot_base.datatypes import GaitConfig, Velocities, Pose, Point, Euler
from robot_base.base import QuadrupedBase
from body_controller.body_control import BodyController
from robot_base.leg import LegController
from kinematics.kinematic import Kinematics

class QuadrupedController:
    def __init__(self, loop_rate: float = 200.0):
        """
        初始化四足机器人总控制器
        :param loop_rate: 控制循环频率 (Hz)，默认 200Hz
        """
        self.loop_rate = loop_rate
        self.is_running = False
        self._thread = None

        # 1. 初始化数据结构
        self.req_vel = Velocities()
        self.req_pose = Pose()
        
        # 2. 初始化步态配置 (替代 ROS 的 getParam)
        self.gait_config = GaitConfig()
        self.gait_config.pantograph_leg = False
        self.gait_config.max_linear_velocity_x = 0.5
        self.gait_config.max_linear_velocity_y = 0.25
        self.gait_config.max_angular_velocity_z = 1.0
        self.gait_config.com_x_translation = 0.0
        self.gait_config.swing_height = 0.04
        self.gait_config.stance_depth = 0.0
        self.gait_config.stance_duration = 0.25
        self.gait_config.nominal_height = 0.20
        self.gait_config.knee_orientation = ">>"  # ">>" 狗腿, "><" 蜘蛛腿

        # 设置默认站立高度
        self.req_pose.position.z = self.gait_config.nominal_height

        # 3. 实例化核心控制组件
        self.base = QuadrupedBase(self.gait_config)
        self.body_controller = BodyController(self.base)
        self.leg_controller = LegController(self.base, self._get_time_us())
        self.kinematics = Kinematics(self.base)

        # 锁定机制：防止外部输入与主循环发生数据竞争
        self._lock = threading.Lock()

        # 用于保存最新计算出的关节角度，供外部读取
        self.current_joint_angles: List[float] = [0.0] * 12

    def _get_time_us(self) -> int:
        """获取当前时间的微秒时间戳 (替代 rosTimeToChampTime)"""
        return int(time.time() * 1_000_000)

    # ==========================================
    # 外部控制接口 (替代 ROS Subscribers)
    # ==========================================
    def set_cmd_vel(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        """设置机器人的目标移动速度"""
        with self._lock:
            self.req_vel.linear.x = linear_x
            self.req_vel.linear.y = linear_y
            self.req_vel.angular.z = angular_z

    def set_cmd_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        """设置机器人的身体姿态偏移"""
        with self._lock:
            self.req_pose.position.x = x
            self.req_pose.position.y = y
            # 加上名义高度作为基准
            self.req_pose.position.z = z + self.gait_config.nominal_height
            
            self.req_pose.orientation.roll = roll
            self.req_pose.orientation.pitch = pitch
            self.req_pose.orientation.yaw = yaw

    def get_joint_angles(self) -> List[float]:
        """获取当前解算出的 12 个电机角度"""
        with self._lock:
            return list(self.current_joint_angles)

    # ==========================================
    # 核心控制循环 (替代 ROS Timer)
    # ==========================================
    def _control_loop(self):
        period = 1.0 / self.loop_rate
        
        while self.is_running:
            loop_start_time = time.time()
            current_time_us = self._get_time_us()

            # 准备 4 个空矩阵存放目标足端位姿
            target_foot_positions = [np.eye(4) for _ in range(4)]

            with self._lock:
                # 1. 身体位姿控制：根据身体姿态要求，计算脚掌相对于大腿根部的初始位移
                self.body_controller.pose_command_all(target_foot_positions, self.req_pose)

                # 2. 腿部步态控制：叠加因为走路（迈步）产生的贝塞尔曲线位移
                self.leg_controller.velocity_command(target_foot_positions, self.req_vel, current_time_us)

                # 3. 逆向运动学解算：将脚掌 3D 坐标转化为 12 个电机的角度
                target_joints = self.kinematics.inverse_all(target_foot_positions)

                if target_joints is not None:
                    self.current_joint_angles = target_joints
                    # 此处相当于 ROS 中的 publishJoints_
                    # 在实际工程中，你可以在这里通过串口将角度发给下位机，或者发送给 PyBullet
                    # print(f"Output Joints: {target_joints}") 
                else:
                    # 解算失败（不可达），丢弃本帧
                    pass

            # 精确控制循环频率 (休眠补齐时间)
            elapsed_time = time.time() - loop_start_time
            sleep_time = period - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """启动后台控制线程"""
        if not self.is_running:
            print("[INFO] Starting Quadruped Controller Thread...")
            self.is_running = True
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """停止后台控制线程"""
        if self.is_running:
            print("[INFO] Stopping Quadruped Controller...")
            self.is_running = False
            self._thread.join()
            print("[INFO] Controller stopped.")


# ==========================================
# 本地测试与执行流程 (替代 ROS node main)
# ==========================================
if __name__ == "__main__":
    # 1. 实例化控制器
    controller = QuadrupedController(loop_rate=200.0)
    
    # 2. 启动控制循环 (将在后台线程运行)
    controller.start()

    try:
        print("\n--- 阶段 1: 原地起立待命 (2秒) ---")
        controller.set_cmd_vel(0.0, 0.0, 0.0)
        time.sleep(2.0)
        
        print("\n--- 阶段 2: 原地身体姿态测试 (俯仰 Pitch) ---")
        # 身体向下压 2cm，抬头(Pitch) 0.2弧度
        controller.set_cmd_pose(0.0, 0.0, -0.02, 0.0, 0.2, 0.0)
        time.sleep(1.0)
        print(f"当前电机角度: {[round(j, 3) for j in controller.get_joint_angles()]}")
        time.sleep(1.0)
        
        print("\n--- 阶段 3: 恢复平躺，准备前进 ---")
        controller.set_cmd_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        time.sleep(1.0)

        print("\n--- 阶段 4: 步态测试 (向前行走 0.2 m/s) ---")
        controller.set_cmd_vel(0.2, 0.0, 0.0)
        
        # 模拟运行并在主线程以 10Hz 频率打印读取到的电机角度
        for i in range(20):
            angles = controller.get_joint_angles()
            print(f"t={i*0.1:.1f}s | LF Hip: {angles[0]:.2f}, LF Up: {angles[1]:.2f}, LF Low: {angles[2]:.2f}")
            time.sleep(0.1)
            
        print("\n--- 阶段 5: 停止 ---")
        controller.set_cmd_vel(0.0, 0.0, 0.0)
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n用户中断运行...")
        
    finally:
        # 3. 清理并退出
        controller.stop()