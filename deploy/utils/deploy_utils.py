import numpy as np
from scipy.spatial.transform import Rotation as R


def modify_state(state): 
    position = state[0:3] / 1000000   # 0.001mm -> 1m 
    rotation = state[3:6] / 1000 * np.pi / 180   # 0.001deg -> rad 
    gripper = state[6:7] / 1000000   # 0.001mm -> 1m 
    return np.concatenate([position, rotation, gripper])


def process_actions(
    state: np.ndarray,
    action: np.ndarray,
    action_type: str,
    *,
    euler_sequence: str = "xyz",
    degrees: bool = False,
) -> np.ndarray:
    """
    将输入的 (state, action, action_type) 规范化为“绝对”的动作序列并返回。
    
    参数
    ----
    state : np.ndarray
        shape 为 (7,) 或 (1, 7)。
        - joint: [j1, j2, j3, j4, j5, j6, gripper]
        - endpose: [x, y, z, r, p, y, gripper]，rpy 顺序由 euler_sequence 指定。
    action : np.ndarray
        shape 为 (n, 7)。
    action_type : str
        6 种之一：
        - 'absolute_joint', 'relative_joint', 'delta_joint'
        - 'absolute_endpose', 'relative_endpose', 'delta_endpose'
    euler_sequence : str, 默认 'xyz'
        rpy 的欧拉角顺序（传给 scipy Rotation）。
    degrees : bool, 默认 False
        若为 True，则 rpy 以角度为单位；否则为弧度。
        
    返回
    ----
    np.ndarray
        shape 为 (n, 7) 的“绝对”动作序列（对 absolute_* 则为原样返回）。
    """
    # --- 形状与拷贝 ---
    state = np.asarray(state).reshape(-1)
    if state.size != 7:
        raise ValueError(f"state must have 7 elements, got shape {state.shape}")
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"action must be (n, 7), got shape {action.shape}")
    n = action.shape[0]
    out = np.empty_like(action, dtype=float)

    # --- 一些工具 ---
    def split_pose(x):
        """将 (k,7) 或 (7,) 切为 pos(3), rpy(3), g(1)"""
        x = np.asarray(x)
        return x[..., :3], x[..., 3:6], x[..., 6]

    def join_pose(pos, rpy, g):
        return np.concatenate([pos, rpy, g[..., None]], axis=-1)

    def rpy_to_rot(rpy):
        return R.from_euler(euler_sequence, rpy, degrees=degrees)

    def rot_to_rpy(rot: R):
        return rot.as_euler(euler_sequence, degrees=degrees)

    # --- 绝对类型：直接返回 ---
    if action_type in ("absolute_joint", "absolute_endpose"):
        return action.copy()

    # --- joint 系列 ---
    if action_type == "relative_joint":
        # 所有 action 直接加 state
        out[:, :6] = action[:, :6] + state[:6]
        out[:, 6] = action[:, 6] + state[6]
        return out

    if action_type == "delta_joint":
        # 累加式：abs0 = state + a0; abs1 = abs0 + a1; ...
        out[:, :6] = state[:6] + np.cumsum(action[:, :6], axis=0)
        out[:, 6] = action[:, 6]
        return out

    # --- endpose 系列 ---
    if action_type == "relative_endpose":
        # 位置、夹爪：逐条加 state；姿态：R_abs = R_state * R_rel_i（逐条独立，不累积）
        s_pos, s_rpy, s_g = split_pose(state)
        R_s = rpy_to_rot(s_rpy)
        a_pos, a_rpy, a_g = split_pose(action)

        # 位置与夹爪
        pos_abs = a_pos + s_pos
        g_abs = a_g + s_g

        # 姿态（对每一条 action 独立相乘）
        R_rel = rpy_to_rot(a_rpy)
        # scipy 支持批量 Rotation，相乘将逐元素相乘
        R_abs = R_s * R_rel
        rpy_abs = rot_to_rpy(R_abs)

        return join_pose(pos_abs, rpy_abs, g_abs)

    if action_type == "delta_endpose":
        # 位置：累加
        # 姿态：从 R_state 开始，逐步左乘增量：R_abs_k = R_abs_{k-1} * R_delta_k
        s_pos, s_rpy, s_g = split_pose(state)
        a_pos, a_rpy, a_g = split_pose(action)

        # 位置 & 夹爪的累加
        pos_abs = s_pos + np.cumsum(a_pos, axis=0)
        # g_abs = s_g + np.cumsum(a_g, axis=0)
        g_abs = a_g
        # g_abs = np.array([0.001 if i < 0.05 else 0.08 for i in g_abs])

        # 姿态累积
        R_abs_list = []
        R_curr = rpy_to_rot(s_rpy)
        # 将所有增量转换为 Rotation（支持批量）
        R_delta_all = rpy_to_rot(a_rpy)
        # 逐步相乘（保持与位置/夹爪的“delta”语义一致）
        for k in range(n):
            R_curr = R_curr * R_delta_all[k]
            # R_curr = R_delta_all[k] * R_curr
            R_abs_list.append(R_curr)
        # print(1c)
        rpy_abs = rot_to_rpy(R.concatenate(R_abs_list))

        return join_pose(pos_abs, rpy_abs, g_abs)

    # 未知类型
    raise ValueError(f"Unknown action_type: {action_type}")
