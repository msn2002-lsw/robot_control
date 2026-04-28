import time
from typing import List

# 假设这个类已经定义好
from robot_base.base_dh import QuadrupedBase

# 定义常量，与 C++ 的 SECONDS_TO_MICROS 对应
SECONDS_TO_MICROS = 1_000_000

class PhaseGenerator:
    def __init__(self, base: QuadrupedBase, current_time: int|None):
        """
        初始化相位生成器。
        
        :param base: QuadrupedBase 实例，用于读取配置（如支撑时长）
        :param current_time: 初始化时的时间戳（微秒）。如果不传则取当前时间。
        """
        self._base: QuadrupedBase = base
        
        if current_time is None:
            self._last_touchdown: int = self.now()
        else:
            self._last_touchdown: int = current_time
            
        self.has_swung: bool = False
        self.has_started: bool = False
        self.has_finished_first_stride: bool = False
        
        # 相当于 C++ 中的 float array[4]
        # 分别存储 LF(0), RF(1), LH(2), RH(3) 的相位信号 (取值 0.0 ~ 1.0)
        self.stance_phase_signal: List[float] = [0.0, 0.0, 0.0, 0.0]
        self.swing_phase_signal: List[float] = [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def now() -> int:
        """
        静态方法，模拟 C++ 宏中的 time_us()
        返回当前时间的微秒时间戳。
        """
        # time.time() 返回以秒为单位的浮点数，乘以一百万后转为整数
        return int(time.time() * SECONDS_TO_MICROS)

    def run(self, target_velocity: float, step_length: float, current_time: int|None) -> None:
        """
        核心运行逻辑：根据流逝的时间和目标速度，计算四条腿的支撑/摆动相位。
        
        :param target_velocity: 目标速度 (其实这里代码只判断了是否为 0)
        :param step_length: 步长 (原代码中并未使用该参数，保留是为了接口一致)
        :param current_time: 当前微秒时间戳
        """
        if current_time is None:
            current_time = self.now()
            
        elapsed_time_ref = 0
        
        # 摆动相时间固定为 0.25 秒（转换为微秒）
        swing_phase_period = 0.25 * SECONDS_TO_MICROS
        
        # 支撑相时间从机器人的 gait_config 中读取（转换为微秒）
        stance_phase_period = self._base.gait_config.stance_duration * SECONDS_TO_MICROS
        
        # 一个完整的跨步周期 = 支撑相 + 摆动相
        stride_period = stance_phase_period + swing_phase_period
        
        leg_clocks = [0.0, 0.0, 0.0, 0.0]

        # ==========================================
        # 1. 速度为 0 时的静止逻辑
        # ==========================================
        if target_velocity == 0.0:
            self._last_touchdown = 0
            self.has_swung = False
            for i in range(4):
                self.stance_phase_signal[i] = 0.0
                self.swing_phase_signal[i] = 0.0
            return

        # ==========================================
        # 2. 状态机与时间流逝更新
        # ==========================================
        if not self.has_started:
            self.has_started = True
            self._last_touchdown = current_time

        # 如果时间已经过了一个完整的周期，重置 touchdown 时间
        if (current_time - self._last_touchdown) >= stride_period:
            self._last_touchdown = current_time

        elapsed_time_ref = current_time - self._last_touchdown
        if elapsed_time_ref >= stride_period:
            elapsed_time_ref = stride_period

        # ==========================================
        #  相位差设定
        # ==========================================
        leg_clocks[0] = elapsed_time_ref - (self._base.gait_config.phase_offset[0] * stride_period) # LF
        leg_clocks[1] = elapsed_time_ref - (self._base.gait_config.phase_offset[1] * stride_period) # RF
        leg_clocks[2] = elapsed_time_ref - (self._base.gait_config.phase_offset[2] * stride_period) # LH
        leg_clocks[3] = elapsed_time_ref - (self._base.gait_config.phase_offset[3] * stride_period) # RH

        # ==========================================
        # 4. 计算 0.0 -> 1.0 的归一化相位信号
        # ==========================================
        for i in range(4):
            # 支撑相计算
            if 0 < leg_clocks[i] < stance_phase_period:
                self.stance_phase_signal[i] = leg_clocks[i] / stance_phase_period
            else:
                self.stance_phase_signal[i] = 0.0

            # 摆动相计算（处理周期循环前后的负值溢出逻辑）
            if -swing_phase_period < leg_clocks[i] < 0:
                self.swing_phase_signal[i] = (leg_clocks[i] + swing_phase_period) / swing_phase_period
            elif stance_phase_period < leg_clocks[i] < stride_period:
                self.swing_phase_signal[i] = (leg_clocks[i] - stance_phase_period) / swing_phase_period
            else:
                self.swing_phase_signal[i] = 0.0

        # ==========================================
        # 5. 起步第一步的特殊平滑处理
        # ==========================================
        # 只要全局时间还没有跑完完整的第一圈，就依然处于“起步寻优”阶段
        if not self.has_finished_first_stride:
            
            for i in range(4):
                # 核心逻辑：如果当前流逝的全局相位，还没达到这条腿预设的起步延迟点
                # 说明这条腿还没到它该动的时候，必须强行抑制！
                if elapsed_time_ref < self._base.gait_config.phase_offset[i] * stride_period:
                    # 钳制在当前物理位置 (具体设为0.0还是保持当前IK坐标，取决于底层控制闭环)
                    self.stance_phase_signal[i] = 0.0
                    self.swing_phase_signal[i] = 0.0
                    
            # 只有当全局相位接近 1.0 时，所有腿才都至少动作过一次
            if elapsed_time_ref >= 0.95 * stride_period:
                self.has_finished_first_stride = True