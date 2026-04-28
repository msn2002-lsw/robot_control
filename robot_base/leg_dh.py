import numpy as np

class Joint:
    """
    标准 DH 关节类

    标准 DH 参数:
        theta: 绕 z_{i-1} 轴旋转角，单位 rad
        d:     沿 z_{i-1} 轴平移距离
        a:     沿 x_i 轴平移距离
        alpha: 绕 x_i 轴旋转角，单位 rad

    注意:
        本类默认使用标准 DH 法，不是改进 DH 法。
    """

    def __init__(
        self,
        theta: float,
        d: float,
        a: float,
        alpha: float,
        joint_type: str = "revolute",
        offset: float = 0.0,
        qlim: tuple | None = None
    ):
        """
        参数说明:
            theta: DH 参数 theta
            d: DH 参数 d
            a: DH 参数 a
            alpha: DH 参数 alpha
            joint_type: 关节类型，可选 "revolute" 或 "prismatic"
            offset: 关节零点偏置，单位 rad 或 m
            qlim: 关节限位，例如 (-1.57, 1.57)
        """

        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha

        self.joint_type = joint_type
        self.offset = offset
        self.qlim = qlim

        if self.joint_type not in ["revolute", "prismatic", "fixed"]:
            raise ValueError("joint_type 只能是 'revolute', 'prismatic' 或 'fixed'")

    def transform(self, q: float = 0.0) -> np.ndarray:
        """
        输出当前关节的标准 DH 齐次变换矩阵

        对于 revolute 关节:
            q 表示关节转角增量

        对于 prismatic 关节:
            q 表示关节伸缩位移增量

        对于 fixed 关节:
            q 不起作用

        返回:
            4x4 numpy.ndarray 齐次变换矩阵
        """

        if self.qlim is not None:
            q_min, q_max = self.qlim
            if not (q_min <= q <= q_max):
                raise ValueError(f"关节变量 q={q} 超出限位范围 {self.qlim}")

        if self.joint_type == "revolute":
            theta = self.theta + q + self.offset
            d = self.d

        elif self.joint_type == "prismatic":
            theta = self.theta
            d = self.d + q + self.offset

        else:  # fixed
            theta = self.theta
            d = self.d

        a = self.a
        alpha = self.alpha

        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,       sa,       ca,      d],
            [0,        0,        0,      1]
        ])

        return T

    def __repr__(self):
        return (
            f"Joint(theta={self.theta}, d={self.d}, a={self.a}, "
            f"alpha={self.alpha}, joint_type='{self.joint_type}', "
            f"offset={self.offset}, qlim={self.qlim})"
        )

if __name__ == "__main__":
    # 示例：一个旋转关节
    joint1 = Joint(
        theta=0.0,
        d=0,
        a=13,
        alpha=np.pi / 2,
        joint_type="revolute",
        offset=0.0,
        qlim=(-np.pi, np.pi)
    )
    joint2 = Joint(
        theta=0.0,
        d=105,
        a=39,
        alpha=0.0,
        joint_type="revolute",
        offset=0.0,
        qlim=(-np.pi, np.pi)
    )
    joint3 = Joint(
        theta=0.0,
        d=0,
        a=28,
        alpha=-np.pi / 2,
        joint_type="revolute",
        offset=0.0,
        qlim=(-np.pi, np.pi)
    )
    joint4 = Joint(
        theta=0.0,
        d=0,
        a=49,
        alpha=0,
        joint_type="revolute",
        offset=0.0,
        qlim=(-np.pi, np.pi)
    )
    joint5 = Joint(
        theta=0.0,
        d=0,
        a=60,
        alpha=-np.pi / 2,
        joint_type="revolute",
        offset=-np.pi / 2,
        qlim=(-np.pi, np.pi)
    )
    joint6 = Joint(
        theta=0.0,
        d=0,
        a=60,
        alpha=0,
        joint_type="revolute",
        offset=np.pi / 2,
        qlim=(-np.pi, np.pi)
    )

    # 设置关节变量
    q1 = np.deg2rad(0)
    q2 = np.deg2rad(0)
    q3 = np.deg2rad(-90)
    q4 = np.deg2rad(0)
    q5 = np.deg2rad(90)
    q6 = np.deg2rad(-90)

    # 计算末端执行器相对于基座的齐次变换矩阵
    T06 = joint1.transform(q1) @ joint2.transform(q2) @ joint3.transform(q3) @ joint4.transform(q4) @ joint5.transform(q5) @ joint6.transform(q6)
    np.set_printoptions(precision=4, suppress=True)
    print("T06 =")
    print(T06)