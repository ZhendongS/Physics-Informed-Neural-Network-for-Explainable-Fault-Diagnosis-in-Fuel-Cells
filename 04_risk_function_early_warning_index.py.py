# -*- coding: utf-8 -*-
"""
基于 F01_output.mat / comprehensive_results 计算风险函数 RF(t)

- 从 MAT 读取 comprehensive_results
- 使用 res, pV, pT, pH, pO 构造 RF(t)
- 在“正常工况”样本上估计 μ, σ
- 原版：在全序列上计算 HI 提前量
- 扩展版：按 “电流 + 某一故障大类（如 220A 水淹）” 分别计算 RF 提前量

说明（本版改动）：
- 保持参数不变
- 不再绘制 C(t)
- 电压报警阈值 = 片段初始电压 - 0.1V，V(t) 低于该阈值判为电压报警
- RF 报警设为两级阈值（预警 / 危险），绘图中按阈值对 RF 轴区域着色
- RF 首次报警时间：RF 曲线第一次大于“第一报警值（预警阈值）”
- 不绘制标题，不绘制图例
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ==================== 统一绘图风格设置（集中配置，便于调试） ====================
# - 分辨率 300 dpi
# - Arial 字体
# - 坐标轴标签字号约 11，刻度字号约 10
# - 图像尺寸单图约 4.5×3 英寸（在具体绘图处使用）
# - 坐标轴与刻度线宽度约 1.0，必要时图内边框可加粗到 1.5
# - 颜色采用固定调色板（供本脚本使用）
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "Arial",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

# 固定调色板（可按需在此集中调整）
COLOR_PALETTE = {
    "voltage_line": "#6d8ef7",      # 电压曲线（蓝）
    "voltage_thresh": "#2b47e8",    # 电压阈值线
    "rf_line": "#e41a1c",           # RF 曲线（红）
    "rf_warn_band": "#a9e9e4",      # RF 低风险区背景
    "rf_mid_band": "#fee695",       # RF 预警区背景
    "rf_danger_band": "#f5b7bf",    # RF 高风险区背景
    "v_alarm_vline": "#377eb8",     # 电压报警竖线
    "rf_alarm_vline": "#e41a1c",    # RF 报警竖线
}

# ========= 与 F02_E09_figure9.py 保持一致的索引 =========
INDEX = {
    **{f"x{i}": i for i in range(8)},
    "y_true": 8, "y_pred": 9, "ale": 10, "epi": 11, "res": 12,
    "pV": 13, "pT": 14, "pH": 15, "pO": 16, "label": 17
}

MAT_PATH = "F01_output.mat"        # 你的 MAT 文件路径

# 假设 x0 是电流列；若不是，请改成正确的列名或直接用索引
CURRENT_COL = "x0"

# 故障大类对应的 label 范围（根据你的实际编码需要调整）
# 这里示例：
#   1–3：水淹
#   4–6：氧饥饿
#   7–9：膜干
#   10–12：氢饥饿
FAULT_RANGE_MAP = {
    "水淹":   range(1, 4),
    "氧饥饿": range(4, 7),
    "膜干":   range(7, 10),
    "氢饥饿": range(10, 13),
}

# ========= RF 相关关键参数（全部集中在此，便于调参） =========

# 用于估计“正常工况”均值和方差的标签
NORMAL_LABELS = (0,)

# RF 使用的残差键
RF_RES_KEYS = ("res", "pV", "pT", "pH", "pO")

# 分层设置（索引将根据 RF_RES_KEYS 顺序自动映射）
# 电压层：["res", "pV"]，气体层：["pH", "pO"]，温度层：["pT"]
RF_LAYER_CONFIG = {
    "voltage": ["res", "pV"],
    "gas":     ["pH", "pO"],
    "temp":    ["pT"],
}

# 各残差权重（顺序与 RF_RES_KEYS 对应）
RF_FEATURE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

# 各层权重（线性叠加）
RF_LAYER_WEIGHTS = {
    "voltage": 1.0,
    "gas":     1.0,
    "temp":    1.0,
}

# 层内 p-范数
RF_P_LAYER = 2.0

# 3σ 安全区阈值（标准差倍数）
RF_Z_SAFE = 2.0

# 时间积分衰减因子 λ
RF_LAMBDA_DECAY = 0.9971

# Logistic 映射参数
RF_K_LOGISTIC = 0.0005      # 斜率
RF_C0_LOGISTIC = 500.0      # 拐点 C0：C=C0 时 Logistic≈0.5
RF_C_MAX = 1000.0           # 认为“极高累计风险”的 C 值，用于归一化到 1

# 电压与 RF 的告警门限（原始参数，保留但不再用于绘图中的固定阈值）
V_THRESHOLD_FULL = 0.7
RF_THRESHOLD_FULL = 0.4

V_THRESHOLD_COND = 3.1
RF_THRESHOLD_COND = 0.4

# 指数平滑系数（如果希望对 RF 再做时间平滑）
RF_ALPHA_SMOOTH = 0.2

# 计算“电流 + 故障大类”时的电流容差
CURRENT_TOL = 0.5

# 想要评估的 “电流 + 故障大类” 组合
RF_CONDITIONS = [
    (108.0, "氢饥饿", (0, 1050)),
    (108.0, "膜干", None),
    (270.0, "氧饥饿", None),
    (270.0, "膜干", None),
    # (108.0, "水淹", None),
    # (270.0, "水淹", None),
    # (405.0, "水淹", None),
]
RF_CONDITIONS = [
    (108.0, "水淹", (0, 1050)),
    (108.0, "氧饥饿", None),
    (108.0, "膜干", None),
    (108.0, "氢饥饿", None),
    (270.0, "水淹", None),
    (270.0, "膜干", None),
    (270.0, "氧饥饿", None),
    (270.0, "氢饥饿", None),
    (405.0, "水淹", None),
    (405.0, "氧饥饿", None),
    (405.0, "膜干", None),
    (405.0, "氢饥饿", None),
    # (108.0, "水淹", None),
    # (270.0, "水淹", None),
    # (405.0, "水淹", None),
]
# ===== RF 两级报警阈值（预警 / 危险） =====
RF_WARN_THRESHOLD = 0.3    # 第一报警值（预警）
RF_DANGER_THRESHOLD = 0.6  # 第二报警值（危险）


# --------- 1. 读取 comprehensive_results ----------
def load_comprehensive_results(mat_path: str) -> np.ndarray:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"未找到文件: {mat_path}")
    data = sio.loadmat(mat_path)
    if "comprehensive_results" not in data:
        raise KeyError("MAT 文件中未找到变量 'comprehensive_results'")
    arr = np.array(data["comprehensive_results"])
    if arr.shape[1] <= max(INDEX.values()):
        raise ValueError(f"数据列数为 {arr.shape[1]}，不足以满足索引布局。")
    return arr


# --------- 2. 在“正常工况”样本上估计 μ, σ ----------
def estimate_mu_sigma_normal(results: np.ndarray,
                             res_keys=RF_RES_KEYS,
                             normal_labels=NORMAL_LABELS):
    labels = results[:, INDEX["label"]].astype(int)
    mask_normal = np.isin(labels, normal_labels)
    if not mask_normal.any():
        raise ValueError(f"数据中未找到正常工况标签 {normal_labels} 的样本。")

    R_normal = np.stack(
        [results[mask_normal, INDEX[k]].astype(float) for k in res_keys],
        axis=1
    )  # [N_normal, D]

    mu = np.nanmean(R_normal, axis=0)
    sigma = np.nanstd(R_normal, axis=0, ddof=1)
    sigma[sigma == 0] = 1e-6
    return mu, sigma


# --------- 3. 计算风险函数 RF(t)（替代原来的 HI） ----------
def compute_rf_time_series(results: np.ndarray,
                           mu: np.ndarray,
                           sigma: np.ndarray,
                           res_keys=RF_RES_KEYS,
                           feature_weights: np.ndarray = RF_FEATURE_WEIGHTS,
                           layer_config: dict = RF_LAYER_CONFIG,
                           layer_weights: dict = RF_LAYER_WEIGHTS,
                           p_layer: float = RF_P_LAYER,
                           z_safe: float = RF_Z_SAFE,
                           lambda_decay: float = RF_LAMBDA_DECAY,
                           k_logistic: float = RF_K_LOGISTIC,
                           C0_logistic: float = RF_C0_LOGISTIC,
                           C_max: float = RF_C_MAX,
                           alpha_smooth: float = RF_ALPHA_SMOOTH):
    """
    返回：
      RF_inst(t), RF_smooth(t), extra
    其中 extra["C"] 即累计风险积分 C(t)
    """
    R = np.stack(
        [results[:, INDEX[k]].astype(float) for k in res_keys],
        axis=1
    )
    N, D = R.shape

    if feature_weights is None:
        w_feat = np.ones(D, dtype=float)
    else:
        w_feat = np.asarray(feature_weights, dtype=float)
        if w_feat.shape[0] != D:
            raise ValueError(f"feature_weights 长度应为 {D}，当前为 {len(w_feat)}")

    # 1) 标准化 + 绝对值
    z = (R - mu.reshape(1, -1)) / sigma.reshape(1, -1)   # [N, D]
    a = np.abs(z)

    # 2) 3σ 安全区截断
    a_trunc = np.maximum(0.0, a - z_safe)                # [N, D]

    # 建立：特征名 -> 在 res_keys 中的列索引
    key_to_idx = {k: i for i, k in enumerate(res_keys)}

    # 3) 每层 p-范数
    S_layers = {}
    for layer_name, key_list in layer_config.items():
        idxs = [key_to_idx[k] for k in key_list if k in key_to_idx]
        if len(idxs) == 0:
            S_layers[layer_name] = np.zeros(N, dtype=float)
            continue
        A_l = a_trunc[:, idxs]                      # [N, Dl]
        W_l = w_feat[idxs].reshape(1, -1)           # [1, Dl]
        S_l = np.power((W_l * np.power(A_l, p_layer)).sum(axis=1), 1.0 / p_layer)
        S_layers[layer_name] = S_l                 # [N]

    # 4) 层间线性加权合成瞬时强度 S_tot(t)
    S_tot = np.zeros(N, dtype=float)
    for lname, S_l in S_layers.items():
        beta = layer_weights.get(lname, 1.0)
        S_tot += beta * S_l

    # 5) 带衰减积分：C(t) = λ C(t-1) + S_tot(t)
    C = np.zeros(N, dtype=float)
    for t in range(1, N):
        C[t] = lambda_decay * C[t - 1] + S_tot[t]

    # 6) Logistic 映射 + 归一化到 [0,1]
    C_clip = np.clip(C, 0.0, C_max)
    L_0 = 1.0 / (1.0 + np.exp(-k_logistic * (0.0 - C0_logistic)))
    L_max = 1.0 / (1.0 + np.exp(-k_logistic * (C_max - C0_logistic)))
    denom = (L_max - L_0) if (L_max - L_0) != 0 else 1e-6

    RF_inst = (1.0 / (1.0 + np.exp(-k_logistic * (C_clip - C0_logistic))) - L_0) / denom
    RF_inst = np.clip(RF_inst, 0.0, 1.0)

    # 7) 对 RF_inst 再做一次时间平滑
    RF_smooth = np.zeros_like(RF_inst)
    RF_smooth[0] = RF_inst[0]
    for t in range(1, N):
        RF_smooth[t] = alpha_smooth * RF_inst[t] + (1.0 - alpha_smooth) * RF_smooth[t - 1]

    return RF_inst, RF_smooth, {
        "S_layers": S_layers,
        "S_tot": S_tot,
        "C": C,          # 仍返回，方便后续可能的分析，但不再绘图
    }


# --------- 4. 查找首次告警采样点 ----------
def find_first_alarm_index(series: np.ndarray,
                           threshold: float,
                           mode: str = "above"):
    if mode == "above":
        idxs = np.where(series >= threshold)[0]
    elif mode == "below":
        idxs = np.where(series <= threshold)[0]
    else:
        raise ValueError("mode 必须为 'above' 或 'below'")
    if len(idxs) == 0:
        return None
    return int(idxs[0])


# --------- 5. 按“电流 + 故障大类”计算 RF 提前量 ----------
def compute_rf_advance_for_condition(
    results: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    fault_name: str,
    current_target: float,
    current_tol: float = CURRENT_TOL,
    res_keys=RF_RES_KEYS,
    feature_weights: np.ndarray = RF_FEATURE_WEIGHTS,
    layer_config: dict = RF_LAYER_CONFIG,
    layer_weights: dict = RF_LAYER_WEIGHTS,
    p_layer: float = RF_P_LAYER,
    z_safe: float = RF_Z_SAFE,
    lambda_decay: float = RF_LAMBDA_DECAY,
    k_logistic: float = RF_K_LOGISTIC,
    C0_logistic: float = RF_C0_LOGISTIC,
    C_max: float = RF_C_MAX,
    alpha_smooth: float = RF_ALPHA_SMOOTH,
    # 保留但内部不再使用固定阈值
    V_THRESHOLD: float = V_THRESHOLD_COND,
    RF_THRESHOLD: float = RF_THRESHOLD_COND,
    plot: bool = True,
    index_range=None,
):
    if fault_name not in FAULT_RANGE_MAP:
        raise ValueError(f"未知故障大类 '{fault_name}'，可选：{list(FAULT_RANGE_MAP.keys())}")

    labels = results[:, INDEX["label"]].astype(int)
    I = results[:, INDEX[CURRENT_COL]].astype(float)
    V = results[:, INDEX["y_true"]].astype(float)

    fault_range = FAULT_RANGE_MAP[fault_name]
    mask_fault = np.isin(labels, list(fault_range))
    mask_current = np.abs(I - current_target) <= current_tol
    mask = mask_fault & mask_current

    if not np.any(mask):
        print(f"[{current_target}A {fault_name}] 未找到满足条件的样本。")
        return None

    idx_all = np.where(mask)[0]
    idx_all = np.sort(idx_all)
    total_len = len(idx_all)

    if index_range is not None:
        start, end = index_range
        if start < 0:
            start = 0
        if end is None or end > total_len:
            end = total_len
        if start >= end:
            print(f"[{current_target}A {fault_name}] index_range {index_range} "
                  f"超出可用长度 {total_len}，无有效点。")
            return None

        idx_all = idx_all[start:end]
        print(f"[{current_target}A {fault_name}] 使用相对区间 [{start}, {end})，"
              f"实际样本数: {len(idx_all)}/{total_len}")
    else:
        print(f"[{current_target}A {fault_name}] 未指定 index_range，使用全部 {total_len} 个点。")

    results_sub = results[idx_all]
    V_sub = V[idx_all]
    t_sub = np.arange(len(V_sub))

    # 计算 RF
    RF_inst_sub, RF_smooth_sub, extra = compute_rf_time_series(
        results=results_sub,
        mu=mu,
        sigma=sigma,
        res_keys=res_keys,
        feature_weights=feature_weights,
        layer_config=layer_config,
        layer_weights=layer_weights,
        p_layer=p_layer,
        z_safe=z_safe,
        lambda_decay=lambda_decay,
        k_logistic=k_logistic,
        C0_logistic=C0_logistic,
        C_max=C_max,
        alpha_smooth=alpha_smooth,
    )

    # ---- 计算动态电压阈值与报警点 ----
    # 2. 电压报警阈值设定为（片段）初始值 - 0.1V，以 V_sub 小于该阈值为报警
    V_threshold_dyn = float(V_sub[0]) - 0.1
    idx_v_alarm = find_first_alarm_index(V_sub, V_threshold_dyn, mode="below")

    # 3. RF 第一报警（预警）：RF > RF_WARN_THRESHOLD
    idx_rf_warn_alarm = find_first_alarm_index(RF_smooth_sub, RF_WARN_THRESHOLD, mode="above")

    print(f"[{current_target}A {fault_name}]")
    print(f"  子序列长度: {len(V_sub)}（原始满足条件总长度: {total_len}）")
    print(f"  动态电压门限 = V(0) - 0.1 = {V_threshold_dyn:.4f}")
    print(f"  RF 预警门限 = {RF_WARN_THRESHOLD}")
    print(f"  RF 危险门限 = {RF_DANGER_THRESHOLD}")
    print(f"  电压首次告警索引(子序列): {idx_v_alarm}")
    print(f"  RF 预警首次告警索引(子序列): {idx_rf_warn_alarm}")

    delta_idx = None
    if idx_v_alarm is not None and idx_rf_warn_alarm is not None:
        delta_idx = idx_v_alarm - idx_rf_warn_alarm
        print(f"  ==> RF 预警比电压提前 {delta_idx} 个采样点告警（正数代表提前）。")
    else:
        print("  某一方未触发告警，无法计算提前量。")

    if plot:
        # 图像尺寸单图约 4.5×3 英寸
        fig, ax1 = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=300)

        # ==== 自适应电压 y 轴范围 ====
        v_min = float(np.nanmin(V_sub))
        v_max = float(np.nanmax(V_sub))
        if v_max > v_min:
            margin = 0.1 * (v_max - v_min)
        else:
            margin = 0.02
        v_min_plot = v_min - 2.0 * margin
        v_max_plot = v_max + margin

        # 左轴：电压
        ax1.plot(t_sub, V_sub, "-", color=COLOR_PALETTE["voltage_line"],
                 linewidth=2, zorder=3)
        ax1.axhline(V_threshold_dyn, color=COLOR_PALETTE["voltage_thresh"],
                    linestyle="--", alpha=1, zorder=3)
        ax1.set_xlabel("Sample index")
        ax1.set_ylabel("Voltage", color=COLOR_PALETTE["voltage_line"])
        ax1.tick_params(axis='y', labelcolor=COLOR_PALETTE["voltage_line"])
        ax1.set_ylim(v_min_plot, v_max_plot)

        # 右轴：RF
        ax2 = ax1.twinx()

        # 背景色：在 RF 轴上按数值区间着色
        ax2.set_ylim(-0.05, 1.05)
        ax2.axhspan(0.0, RF_WARN_THRESHOLD,
                    facecolor=COLOR_PALETTE["rf_warn_band"], alpha=0.5, zorder=0)
        ax2.axhspan(RF_WARN_THRESHOLD, RF_DANGER_THRESHOLD,
                    facecolor=COLOR_PALETTE["rf_mid_band"], alpha=0.5, zorder=0)
        ax2.axhspan(RF_DANGER_THRESHOLD, 1.0,
                    facecolor=COLOR_PALETTE["rf_danger_band"], alpha=0.5, zorder=0)

        ax2.plot(t_sub, RF_smooth_sub, "-", color=COLOR_PALETTE["rf_line"],
                 linewidth=2, zorder=4)
        ax2.axhline(RF_WARN_THRESHOLD, color=COLOR_PALETTE["rf_line"], linestyle="--",
                    alpha=0.9, linewidth=1.2, zorder=4)
        # ax2.axhline(RF_DANGER_THRESHOLD, color=COLOR_PALETTE["rf_line"], linestyle="--",
        #             alpha=0.9, linewidth=1.2, zorder=4)
        ax2.set_ylabel("Risk Function (RF)", color=COLOR_PALETTE["rf_line"])
        ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE["rf_line"])

        # 坐标轴边框可适当加粗到 1.5（仅图内边框，加粗以增强可视性）
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        # 标注电压报警时刻
        if idx_v_alarm is not None:
            ax1.axvline(t_sub[idx_v_alarm],
                        color=COLOR_PALETTE["v_alarm_vline"],
                        linestyle=":", alpha=0.8, zorder=5)

        # 标注 RF 预警时刻（第一次超过 RF_WARN_THRESHOLD）
        if idx_rf_warn_alarm is not None:
            ax2.axvline(t_sub[idx_rf_warn_alarm],
                        color=COLOR_PALETTE["rf_alarm_vline"],
                        linestyle=":", alpha=0.8, zorder=5)

        # 按要求：不绘制标题，不绘制图例
        plt.tight_layout()
        plt.show()

    return delta_idx


if __name__ == "__main__":

    print(f"加载数据: {MAT_PATH}")
    results = load_comprehensive_results(MAT_PATH)

    mu, sigma = estimate_mu_sigma_normal(results,
                                         res_keys=RF_RES_KEYS,
                                         normal_labels=NORMAL_LABELS)
    print("各残差 μ:", dict(zip(RF_RES_KEYS, mu)))
    print("各残差 σ:", dict(zip(RF_RES_KEYS, sigma)))

    results_summary = {}

    for cond in RF_CONDITIONS:
        if len(cond) == 2:
            current_target, fault_name = cond
            index_range = None
        elif len(cond) == 3:
            current_target, fault_name, index_range = cond
        else:
            raise ValueError(f"RF_CONDITIONS 项格式错误: {cond}")

        delta = compute_rf_advance_for_condition(
            results=results,
            mu=mu,
            sigma=sigma,
            fault_name=fault_name,
            current_target=current_target,
            current_tol=CURRENT_TOL,
            res_keys=RF_RES_KEYS,
            feature_weights=RF_FEATURE_WEIGHTS,
            layer_config=RF_LAYER_CONFIG,
            layer_weights=RF_LAYER_WEIGHTS,
            p_layer=RF_P_LAYER,
            z_safe=RF_Z_SAFE,
            lambda_decay=RF_LAMBDA_DECAY,
            k_logistic=RF_K_LOGISTIC,
            C0_logistic=RF_C0_LOGISTIC,
            C_max=RF_C_MAX,
            alpha_smooth=RF_ALPHA_SMOOTH,
            V_THRESHOLD=V_THRESHOLD_COND,     # 仅形式上传入，函数内部使用动态阈值
            RF_THRESHOLD=RF_THRESHOLD_COND,   # 保留参数但不再用于报警计算
            plot=True,
            index_range=index_range,
        )
        results_summary[(current_target, fault_name, str(index_range))] = delta

    print("\n===== 汇总：RF 提前量（单位：采样点，正数表示 RF 比电压提前） =====")
    for (current_target, fault_name, idx_range), delta in results_summary.items():
        print(f"{current_target}A {fault_name} {idx_range}: {delta}")