import numpy as np
import math
from robot_base.datatypes import Velocities
from robot_base.leg_dh import QuadrupedLeg
from robot_base.base_dh import QuadrupedBase
from robot_base.datatypes import GaitConfig
from leg_controller.phase_generator import PhaseGenerator
from leg_controller.trajectory_planner import TrajectoryPlanner
from leg_controller.leg_controller import LegController
from robot_base.mat_tool import inverse_transform
from kinematics.kinematic import Kinematics

import matplotlib.pyplot as plt

# 定义时间转换宏
SECONDS_TO_MICROS = 1_000_000

def run_trajectory_simulation(base: QuadrupedBase, phase_gen: PhaseGenerator):
    
    # 为四条腿分别实例化 Leg 和 Planner
    planners = [TrajectoryPlanner(base.legs[i]) for i in range(4)]
    
    # 仿真参数
    # 周期为0.5s，为了看到起步 + 2个完整周期，仿真运行 1.3 秒
    sim_duration = 1 
    dt = 0.002  # 2ms 控制步长
    current_time_us = 0
    step_length = 0.15  # 步长 15cm
    
    # 用于记录数据的字典：每条腿记录 X, Z 坐标和所属状态
    traj_data = {
        0: {'x': [], 'z': [], 'state': []},
        1: {'x': [], 'z': [], 'state': []},
        2: {'x': [], 'z': [], 'state': []},
        3: {'x': [], 'z': [], 'state': []}
    }

    print("正在解算笛卡尔空间轨迹...")
    while current_time_us <= sim_duration * SECONDS_TO_MICROS:
        # 1. 更新相位信号
        phase_gen.run(target_velocity=1.0, step_length=step_length, current_time=current_time_us)
        
        # 2. 对每条腿生成轨迹
        for i in range(4):
            # 构建腿部的初始标称位置矩阵 (这里简化为 Z轴处于地面的单位阵)
            foot_pos = np.eye(4)
            foot_pos[2, 3] = -base.gait_config.nominal_height # Z = -0.25m
            
            stance_sig = phase_gen.stance_phase_signal[i]
            swing_sig  = phase_gen.swing_phase_signal[i]
            
            # 生成轨迹 (就地修改 foot_pos 矩阵)
            planners[i].generate(
                foot_position=foot_pos,
                step_length=step_length,
                rotation=0.0, # 直行，无偏航
                swing_phase_signal=swing_sig,
                stance_phase_signal=stance_sig
            )
            
            # 提取笛卡尔坐标 (相对于机身中心的真实空间坐标)
            x_coord = foot_pos[0, 3]
            z_coord = foot_pos[2, 3]
            
            # 记录状态系用于绘图着色
            if stance_sig > 0:
                state = 'stance'
            elif swing_sig > 0:
                state = 'swing'
            else:
                state = 'suppress' # 起步被抑制状态
                
            traj_data[i]['x'].append(x_coord)
            traj_data[i]['z'].append(z_coord)
            traj_data[i]['state'].append(state)
            
        current_time_us += int(dt * SECONDS_TO_MICROS)

    # ==========================================
    # 4. 可视化 D型 足端轨迹
    # ==========================================
    print("生成四足轨迹图表...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Cartesian Foot Trajectory (Startup to 2nd Cycle)", fontsize=16, fontweight='bold')
    
    titles = ['LF (Left Front)', 'RF (Right Front)', 'LH (Left Hind)', 'RH (Right Hind)']
    
    for i, ax in enumerate(axs.flatten()):
        x = np.array(traj_data[i]['x'])
        z = np.array(traj_data[i]['z'])
        states = np.array(traj_data[i]['state'])
        
        # 根据状态分别绘制线段
        stance_mask = (states == 'stance')
        swing_mask = (states == 'swing')
        suppress_mask = (states == 'suppress')
        
        # 画点或线段 (此处使用 scatter 以展示密度和速度变化)
        ax.scatter(x[stance_mask], z[stance_mask], c='#4CAF50', s=10, label='Stance (Support)')
        ax.scatter(x[swing_mask], z[swing_mask], c='#2196F3', s=10, label='Swing (Air)')
        if np.any(suppress_mask):
            ax.scatter(x[suppress_mask], z[suppress_mask], c='gray', s=30, marker='x', label='Suppressed')

        # 标出起点和地面参考线
        ax.axhline(y=-base.gait_config.nominal_height, color='black', linestyle='--', alpha=0.3, label='Ground Level')
        
        ax.set_title(titles[i])
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.axis('equal') # 保持 X 和 Z 轴比例一致，呈现真实几何形状
        ax.grid(True, linestyle=':', alpha=0.7)
        if i == 1:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.show()

def run_phase_simulation(base: QuadrupedBase, phase_gen: PhaseGenerator):
    # 这里可以实现对 PhaseGenerator 的单独测试，记录并绘制相位信号随时间的变化
    # 🚨 修正 4：传入初始时间 0，确保图表从 t=0 开始，而不是巨大的系统时间戳

    # ==========================================
    # 3. 仿真循环与数据采集
    # ==========================================
    sim_duration = 2.0  # 模拟运行 2.0 秒
    dt = 0.005          # 仿真步长 5ms (200Hz 控制频率)
    current_time_us = 0
    target_vel = 0.5    # 目标速度大于 0，触发起步
    
    # 日志容器
    time_log = []
    phase_logs = {0: [], 1: [], 2: [], 3: []} 

    print("开始模拟运行并采集相位数据...")
    while current_time_us <= sim_duration * SECONDS_TO_MICROS:
        # 显式传入 current_time_us 推进仿真时间
        phase_gen.run(target_velocity=target_vel, step_length=0.1, current_time=current_time_us)
        
        # 记录真实秒数用于画图横轴
        time_sec = current_time_us / SECONDS_TO_MICROS
        time_log.append(time_sec)
        
        # 将分离的支撑相和摆动相信号“拼接”成一个连续的 0~2 的轨迹图
        for i in range(4):
            stance_val = phase_gen.stance_phase_signal[i]
            swing_val = phase_gen.swing_phase_signal[i]
            
            if stance_val > 0:
                unified_phase = stance_val        # 0.0 -> 1.0 代表支撑相
            elif swing_val > 0:
                unified_phase = 1.0 + swing_val   # 1.0 -> 2.0 代表摆动相
            else:
                unified_phase = 0.0               # 起步抑制状态
                
            phase_logs[i].append(unified_phase)
            
        current_time_us += int(dt * SECONDS_TO_MICROS)

    # ==========================================
    # 4. 可视化绘图
    # ==========================================
    print("生成相位时序图表...")
    plt.figure(figsize=(12, 6))
    
    colors = ['#FF4B4B', '#4CAF50', '#2196F3', '#FFC107']
    labels = ['LF (Left Front)', 'RF (Right Front)', 'LH (Left Hind)', 'RH (Right Hind)']
    
    for i in range(4):
        plt.plot(time_log, phase_logs[i], label=labels[i], color=colors[i], linewidth=2.5, alpha=0.8)

    # 填充背景色区分物理状态
    plt.axhspan(0.0, 1.0, facecolor='#E8F5E9', alpha=0.6, label='Stance Phase (支撑相)')
    plt.axhspan(1.0, 2.0, facecolor='#E3F2FD', alpha=0.6, label='Swing Phase (摆动相)')

    # 图表装饰与标注
    plt.title("Quadruped Phase Generator Simulation (Trot Gait)", fontsize=16, fontweight='bold')
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Unified Phase \n (0.0~1.0: Stance | 1.0~2.0: Swing)", fontsize=12)
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.xlim(0, sim_duration)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 标注第一步的抑制区间
    plt.axvline(x=0.25, color='gray', linestyle='-.', alpha=0.7)
    plt.text(0.12, 1.8, 'Startup Suppression\n(RF/LH Locked)', fontsize=10, ha='center', color='gray')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()
    pass

def test_leg_controller(base: QuadrupedBase, controller: LegController, req_vel: Velocities):
    
    # 初始化四条腿的足端位姿矩阵数组
    foot_positions = [np.eye(4) for _ in range(4)]
    
    # 仿真参数
    sim_duration = 2.0
    dt = 0.005          
    current_time_us = 0
    
    traj_data = {i: {'x': [], 'y': [], 'z': []} for i in range(4)}

    print("启动顶层控制器仿真...")
    while current_time_us <= sim_duration * SECONDS_TO_MICROS:
        
        # 【🚀 核心修复点】：在每一次调用业务代码前，提供干净的初始状态！
        # 这样底层规划器的 += 操作就是在固定的物理零位上加偏移，彻底消除积分爆炸。
        for i in range(4):
            foot_positions[i] = base.legs[i].zero_stance()
        
        # 将速度指令下发给控制器
        controller.velocity_command(foot_positions, req_vel, current_time_us)
        
        # 记录 3D 轨迹系以便绘图
        for i in range(4):
            traj_data[i]['x'].append(foot_positions[i][0, 3])
            traj_data[i]['y'].append(foot_positions[i][1, 3])
            traj_data[i]['z'].append(foot_positions[i][2, 3])
            
        current_time_us += int(dt * SECONDS_TO_MICROS)

    # ==========================================
    # 3. 绘图：验证 Raibert 落脚点与轨迹生成
    # ==========================================
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"LegController Simulation (Vx={req_vel.linear.x}m/s, Vy={req_vel.linear.y}m/s)", fontsize=16)

    # 绘制顶视图 (X-Y)
    ax1 = fig.add_subplot(121)
    colors = ['#FF4B4B', '#4CAF50', '#2196F3', '#FFC107']
    labels = ['LF', 'RF', 'LH', 'RH']
    
    for i in range(4):
        x = np.array(traj_data[i]['x'])
        y = np.array(traj_data[i]['y'])
        ax1.plot(x, y, color=colors[i], label=labels[i], alpha=0.8, linewidth=2)
        ax1.scatter(x[0], y[0], marker='x', color='black', s=60, zorder=5) # 起点

    ax1.set_title("Top-Down View (X-Y Plane)")
    ax1.set_xlabel("X Position (m) - Forward")
    ax1.set_ylabel("Y Position (m) - Lateral")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # 绘制侧视图 (X-Z)
    ax2 = fig.add_subplot(122)
    for i in range(4):
        x = np.array(traj_data[i]['x'])
        z = np.array(traj_data[i]['z'])
        ax2.plot(x, z, color=colors[i], label=labels[i], alpha=0.8, linewidth=2)

    ax2.axhline(y=-base.gait_config.nominal_height, color='black', linestyle='--', alpha=0.5, label='Ground')
    ax2.set_title("Side View (X-Z Plane)")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Z Position (m) - Height")
    ax2.axis('equal') # 保持真实物理比例
    ax2.grid(True, linestyle=':')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ==========================================
    # 1. 机器人基础几何学配置
    # ==========================================
    l1, l2, l3, l4, l5, l6, l7 = 0.105, 0.013, 0.039, 0.028, 0.049, 0.060, 0.060
    base = QuadrupedBase()
    base.setup_robot_geometry(l1, l2, l3, l4, l5, l6, l7)
    np.set_printoptions(precision=4, suppress=True)

    # ==========================================
    # 2. 步态配置设定 (修复了致命参数)
    # ==========================================
    # 🚨 修正 1：实例名改为小写 gait_config，防止遮蔽 GaitConfig 类
    gait_config = GaitConfig(
        knee_orientation = ">>",             
        pantograph_leg = False,             
        odom_scaler = 1.0,                   
        max_linear_velocity_x = 1.0,         
        max_linear_velocity_y = 0.5,         
        max_angular_velocity_z = 1.0,        
        com_x_translation = 0.0,             
        swing_height = 0.04,                 
        stance_depth = 0.01,                  
        # 🚨 修正 2：必须大于0，设为 0.25s (结合固定摆动相 0.25s，完整周期为 0.5s)
        stance_duration = 0.75,              
        nominal_height = 0.06,              
        # 🚨 修正 3：改为对角步态 Trot 配置，以便能在图表上看出交替效果
        phase_offset = [0.0, 0.25, 0.5, 0.75]  
    ) 
    base.set_gait_config(gait_config)
    phase_gen = PhaseGenerator(base, current_time=0)
    legcontrol = LegController(base, current_time=0)  # 初始化控制器，虽然这里不直接调用它的方法，但它会设置好内部状态

    # 初始化目标速度指令
    req_vel = Velocities()
    
    # 测试用例：斜向行走 (X=0.2m/s, Y=0.1m/s)，不带偏航旋转
    req_vel.linear.x = 0.2
    req_vel.linear.y = 0.0
    req_vel.angular.z = 0.0 

    a=np.array([0.0, 0.0, 0.0, 1.0])

    T = base.legs[3].foot_from_spine()  # 获取右后腿足端相对于基座的变换矩阵 
    print("右后腿足端相对于基座的变换矩阵 T:")
    print(T)
    angle = Kinematics.inverse_single(base.legs[3], T)
    print(angle)
