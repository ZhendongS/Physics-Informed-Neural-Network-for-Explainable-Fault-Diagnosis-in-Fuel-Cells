## -*- coding: utf-8 -*-
"""
D07_fault_diagnosis_figure6＆7.py
- 新布局分类（18列）
- 逻辑回归故障分类 -> 改为 GMM + 标签后验映射的无监督方法
- 增加：在“某一故障条件（仅取该故障的样本）”下，绘制四类故障概率随索引变化（同 E08 宽度，一半高度、无图例）
- 去除命令行参数，改为顶部 CONFIG 配置
"""

import os
import re
import warnings
import numpy as np
import scipy.io as sio
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression   # 保留导入以兼容旧环境（未使用）
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances        # 如后续想切换 KMeans 可直接用
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====================== CONFIG（请在此处修改参数） ======================
MAT_PATH = "F01_output.mat"          # D06_output.mat 路径
DEFAULT_GROUP_SPEC = "水淹:1,2,3,|氧饥饿:4,5,6,|膜干:7,8,9,|氢饥饿:10,11,12"
DEFAULT_FEATURES = "pV,pT,pH,pO"
TEST_SIZE = 0.25
RANDOM_STATE = 42
USE_BALANCED = True          # 保留此参数，但在 GMM 方案中不再起作用（仅为兼容）
SHOW_COEF_TOPN = 0           # 不打印系数解释（保留）

# 概率图尺寸与样式（E08 同宽，一半高度；散点、图例在下方）
PROB_FIG_SIZE = (12, 4)
PROB_MARKER = "o"
PROB_MARKERSIZE = 20
PROB_ALPHA = 0.75

# 诊断概率配色（与 E08 不同，避免混淆）
DIAG_COLORS = {
    "水淹":   "#e377c2",  # 洋红
    "氧饥饿": "#ff7f0e",  # 橙
    "膜干":   "#17becf",  # 青
    "氢饥饿": "#9464b8"   # 棕
}
PRED_LABEL_TO_NAME = {
    0: "水淹",
    1: "氧饥饿",
    2: "膜干",
    3: "氢饥饿",
}
# 选择“按哪一种故障条件绘图”（仅取该故障的样本绘图）
# 可选：["水淹","氧饥饿","膜干","氢饥饿"]
PLOT_CONDITION_FAULT = "氢饥饿"
SAVE_PROB_FIG = False
PROB_FIG_OUT = "fig_out/D07_prob_under_fault.png"
# ====================== CONFIG END ======================


# 设置中文字体支持
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception:
        try:
            from matplotlib import font_manager
            _ = font_manager.findfont(font_manager.FontProperties(family=['sans-serif']))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            return False
        except Exception:
            return False


_ = setup_chinese_font()

# ========== 新布局索引 ==========
INDEX = {
    **{f"x{i}": i for i in range(8)},
    "y_true": 8, "y_pred": 9, "ale": 10, "epi": 11, "res": 12,
    "pV": 13, "pT": 14, "pH": 15, "pO": 16, "label": 17
}
REQUIRED_MAX_INDEX = max(INDEX.values())  # 17

# --------- 数据加载 ----------
def load_comprehensive_results(mat_path: str) -> np.ndarray:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"未找到文件: {mat_path}")
    data = sio.loadmat(mat_path)
    if "comprehensive_results" not in data:
        raise KeyError("MAT 文件中未找到变量 'comprehensive_results'")
    arr = np.array(data["comprehensive_results"])
    if arr.shape[1] <= REQUIRED_MAX_INDEX:
        raise ValueError(f"数据列数为 {arr.shape[1]}，不足以满足新布局（需要 > {REQUIRED_MAX_INDEX}）。")
    return arr

# --------- 工具函数 ----------
def list_available_features() -> List[str]:
    return sorted(INDEX.keys(), key=lambda k: INDEX[k])


def normalize_feature_spec(spec: str) -> str:
    s = spec.strip()
    s = re.sub(r"[，、；;|]+", ",", s)
    s = re.sub(r"(\d+)\.(\d+)", r"\1,\2", s)
    s = re.sub(r"\s+", ",", s)
    s = re.sub(r",+", ",", s)
    return s.strip(", ")


def parse_features(spec: str) -> List[int]:
    cleaned = normalize_feature_spec(spec)
    tokens = [t for t in cleaned.split(",") if t != ""]
    indices: List[int] = []
    for t in tokens:
        if t.isdigit():
            indices.append(int(t))
        else:
            if t not in INDEX:
                raise KeyError(f"未知特征名称: '{t}'。可用名称: {list_available_features()}")
            indices.append(INDEX[t])
    seen = set()
    ordered = []
    for idx in indices:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    if INDEX["label"] in ordered:
        raise ValueError("不允许将 'label' 作为输入特征。")
    return ordered


def parse_group_spec(spec: str) -> Dict[str, List[int]]:
    s = spec.strip()
    parts = re.split(r"[|；;]\s*|\n+", s)
    groups: Dict[str, List[int]] = {}
    for p in parts:
        if not p.strip():
            continue
        if ":" not in p:
            raise ValueError(f"无法解析分组片段（缺少冒号）: '{p}'")
        name, ids_str = p.split(":", 1)
        name = name.strip()
        ids_str = normalize_feature_spec(ids_str)
        id_tokens = [t for t in ids_str.split(",") if t != ""]
        det_ids: List[int] = []
        for tok in id_tokens:
            if not re.match(r"^-?\d+$", tok):
                raise ValueError(f"非法标签ID: '{tok}'（必须为整数）")
            det_ids.append(int(tok))
        if name in groups:
            raise ValueError(f"重复的组名: '{name}'")
        groups[name] = det_ids
    if not groups:
        raise ValueError("未解析到任何分组。")
    return groups


def build_label_mapper(groups: Dict[str, List[int]]) -> Tuple[Dict[int, int], List[str]]:
    class_names = list(groups.keys())
    detailed_to_coarse: Dict[int, int] = {}
    for coarse_idx, name in enumerate(class_names):
        for det in groups[name]:
            if det in detailed_to_coarse:
                prev = class_names[detailed_to_coarse[det]]
                raise ValueError(f"细分标签 {det} 被多个组包含：'{prev}' 与 '{name}'")
            detailed_to_coarse[det] = coarse_idx
    return detailed_to_coarse, class_names


def extract_X_y(results: np.ndarray, feature_indices: List[int], label_map: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    X_all = results[:, feature_indices].astype(np.float64)
    detailed_labels = results[:, INDEX["label"]].astype(np.int32)
    keep_mask = np.array([det in label_map for det in detailed_labels], dtype=bool)
    if not keep_mask.any():
        raise ValueError("筛选后无样本。请检查分组定义。")
    X = X_all[keep_mask]
    det = detailed_labels[keep_mask]
    y = np.array([label_map[int(d)] for d in det], dtype=np.int32)
    mask_ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask_ok], y[mask_ok]


# --------- （保留但不再使用的）监督分类器构建函数 ----------
def build_classifier(balanced: bool = False) -> Pipeline:
    """
    为保持与旧版本接口兼容，保留此函数。
    现在主流程中不再调用该函数，而是采用 GMM 无监督方法。
    """
    class_weight = "balanced" if balanced else None
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight=class_weight
        ))
    ])
    return clf


# --------- 概率曲线（在“某一故障条件”下，仅取该故障样本绘制四类概率） ----------
def plot_fault_probabilities_for_single_fault(
    results: np.ndarray,
    class_names: list,
    y_prob: np.ndarray,
    test_indices_in_full: np.ndarray,
    fault_name_cn: str,
    save_dir: str = None
):
    """
    在某一种“真实故障条件”的测试样本上绘制四类故障概率（单独一张图）。

    参数:
    - results: 全量 comprehensive_results
    - class_names: 合并类别名（顺序与 y_prob 列一致），如 ["水淹","氧饥饿","膜干","氢饥饿"]
    - y_prob: 测试集样本概率 [n_te, n_classes]
    - test_indices_in_full: 测试集样本在原始 results 中的行索引 [n_te]
    - fault_name_cn: 中文故障名，用来确定细分标签范围，必须是 class_names 里的一个
    - save_dir: 若不为 None，则将图保存在该目录下，文件名中带英文故障名；若为 None，则直接 plt.show()
    """
    # 全量的细分标签
    labels_full = results[:, INDEX["label"]].astype(int)

    # 细分标签范围（与 D06 约定一致）
    RANGE_MAP = {
        "水淹":   range(1, 4),
        "氧饥饿": range(4, 7),
        "膜干":   range(7, 10),
        "氢饥饿": range(10, 13)
    }

    # 中文 -> 英文标题和文件名映射
    EN_TITLE = {
        "水淹": "Flooding",
        "氧饥饿": "Oxygen starvation",
        "膜干": "Membrane drying",
        "氢饥饿": "Hydrogen starvation"
    }
    EN_SHORT = {
        "水淹": "flooding",
        "氧饥饿": "oxygen_starvation",
        "膜干": "membrane_drying",
        "氢饥饿": "hydrogen_starvation"
    }

    if fault_name_cn not in RANGE_MAP:
        print(f"警告：未知的故障条件 '{fault_name_cn}'，跳过绘图。")
        return

    det_range = set(RANGE_MAP[fault_name_cn])

    # 在测试样本中筛选出该真实故障条件的样本
    cond_mask_test = np.array(
        [labels_full[idx] in det_range for idx in test_indices_in_full],
        dtype=bool
    )

    if not cond_mask_test.any():
        print(f"警告：测试集中未找到 '{fault_name_cn}' 样本，跳过绘图。")
        return

    y_prob_cond = y_prob[cond_mask_test]
    x = np.arange(len(y_prob_cond))

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 2.0,
        "grid.alpha": 0.5,
    })

    fig, ax = plt.subplots(1, 1, figsize=PROB_FIG_SIZE, dpi=300)

    # 画四类概率散点
    n_classes = len(class_names)
    for j, cls_name in enumerate(class_names):
        color = DIAG_COLORS.get(cls_name, plt.cm.Dark2(j % 8))
        ax.scatter(
            x, y_prob_cond[:, j],
            s=PROB_MARKERSIZE,
            c=[color],
            alpha=PROB_ALPHA,
            marker=PROB_MARKER,
            edgecolors='none',
            label=cls_name
        )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls=":", alpha=0.5)

    # 标题用英文
    title_en = EN_TITLE.get(fault_name_cn, fault_name_cn)
    ax.set_title(title_en)

    # 图例放在下方
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     handles, labels,
    #     loc="lower center",
    #     ncol=n_classes,
    #     frameon=False,
    #     bbox_to_anchor=(0.5, -0.25)
    # )

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        short_en = EN_SHORT.get(fault_name_cn, fault_name_cn)
        out_path = os.path.join(save_dir, f"prob_{short_en}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"已保存: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_fault_probabilities_for_all_faults_separate(
    results: np.ndarray,
    class_names: list,
    y_prob: np.ndarray,
    test_indices_in_full: np.ndarray,
    save_dir: str = None
):
    """
    对“四种真实故障条件”分别调用一次 plot_fault_probabilities_for_single_fault，
    得到四张单独的图。

    - results, class_names, y_prob, test_indices_in_full 含义同上
    - save_dir: 若不为 None，则四张图都保存在该目录下
    """
    # 这里按 class_names 的顺序来绘制，通常是 ["水淹","氧饥饿","膜干","氢饥饿"]
    for fault_name_cn in class_names:
        plot_fault_probabilities_for_single_fault(
            results=results,
            class_names=class_names,
            y_prob=y_prob,
            test_indices_in_full=test_indices_in_full,
            fault_name_cn=fault_name_cn,
            save_dir=save_dir
        )


# --------- 利用 GMM + 标签后验映射的无监督多类概率估计 ----------
def fit_gmm_and_get_probabilities(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    random_state: int = 42,
    n_components: int = None
) -> Tuple[np.ndarray, np.ndarray, GaussianMixture, np.ndarray]:
    """
    使用 GaussianMixture 对 X_tr 进行无监督建模，
    然后用 y_tr 对每个 GMM 成分进行“标签后验标定”，
    最终对 X_te 给出每类故障的概率 y_prob，以及对应的 y_pred。

    返回:
    - y_prob: [n_te, n_classes]，每个样本属于每个故障类别的概率
    - y_pred: [n_te,]，按最大概率得到的预测类别索引
    - gmm: 拟合好的 GaussianMixture 模型
    - comp_fault_prob: [n_components, n_classes]，P(fault | component)
    """
    if n_components is None:
        # 默认：成分数 = 类别数；你也可以改大一些，比如 2 * n_classes
        n_components = n_classes

    # 1) 在训练集上拟合 GMM（完全无监督）
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state
    )
    gmm.fit(X_tr)

    # 2) 训练集的成分后验概率：resp_tr[i, c] = P(component=c | x_i)
    resp_tr = gmm.predict_proba(X_tr)   # [n_tr, n_components]

    # 3) 利用 y_tr 统计 P(fault=k | component=c)
    comp_fault_prob = np.zeros((n_components, n_classes), dtype=float)

    for c in range(n_components):
        weights_c = resp_tr[:, c]   # 该成分在各样本上的后验权重
        total_weight_c = weights_c.sum()
        if total_weight_c <= 0:
            # 若该成分几乎无人“使用”，则给一个均匀分布兜底
            comp_fault_prob[c, :] = 1.0 / n_classes
            continue

        for k in range(n_classes):
            mask_k = (y_tr == k)
            # 该成分在“真实为 k 类样本”上的加权权重和
            comp_fault_prob[c, k] = weights_c[mask_k].sum()

        s = comp_fault_prob[c, :].sum()
        if s > 0:
            comp_fault_prob[c, :] /= s
        else:
            comp_fault_prob[c, :] = 1.0 / n_classes

    # 4) 在测试集上：先算 P(component | x)，再映射为 P(fault | x)
    resp_te = gmm.predict_proba(X_te)              # [n_te, n_components]
    y_prob = resp_te @ comp_fault_prob            # [n_te, n_classes]
    # 数值上有可能出现非常小的负数/非精确和，做一下裁剪和归一化
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    # 5) 按最大概率得到预测标签
    y_pred = y_prob.argmax(axis=1)

    return y_prob, y_pred, gmm, comp_fault_prob
from sklearn.manifold import TSNE

def plot_tsne_of_test_samples(
    X_te: np.ndarray,
    y_pred: np.ndarray,
    save_path: None
):
    """
    使用 t-SNE 将测试集样本 X_te 降到 2 维，并根据诊断结果 y_pred 上色。
    颜色使用 DIAG_COLORS，对应 "水淹","氧饥饿","膜干","氢饥饿"。

    参数
    ----
    X_te : [n_te, n_features]
        测试集特征（与送入 GMM 的特征保持一致，建议用同一份预处理后的 X_te）
    y_pred : [n_te]
        测试集诊断结果（整数标签：0,1,2,3），会通过 PRED_LABEL_TO_NAME 映射为中文类名
    save_path : str or None
        若提供，则保存图片到该路径；否则直接 plt.show()
    """
    # 将整数标签映射为中文类别名
    label_names = np.array([PRED_LABEL_TO_NAME[int(lbl)] for lbl in y_pred])

    print("Running t-SNE on test samples ...")
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_iter=1000,
        verbose=1,
    )
    X_embedded = tsne.fit_transform(X_te)   # [n_te, 2]

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.8,
        "grid.alpha": 0.4,
    })

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    CLASS_NAME_EN = {
        "水淹": "Flooding",
        "氧饥饿": "Oxygen starvation",
        "膜干": "Membrane drying",
        "氢饥饿": "Hydrogen starvation",
    }
    # 按四个诊断结果分别绘制（仍然用中文作为内部标签）
    class_order = ["水淹", "氧饥饿", "膜干", "氢饥饿"]
    for cname in class_order:
        mask = (label_names == cname)
        if not np.any(mask):
            continue
        color = DIAG_COLORS.get(cname, "#000000")
        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            s=18,
            c=color,
            alpha=0.8,
            label=CLASS_NAME_EN.get(cname, cname),  # 图例显示英文
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title("t-SNE of test samples colored by diagnosed fault type")

    #ax.legend(loc="best", frameon=False)

    plt.tight_layout()

    plt.show()

# ---------------- 主流程 ----------------


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # 读取数据
    print(f"加载数据: {MAT_PATH}")
    results = load_comprehensive_results(MAT_PATH)

    # 解析特征与分组
    feature_indices = parse_features(DEFAULT_FEATURES)
    print(f"特征列: {feature_indices}")
    groups = parse_group_spec(DEFAULT_GROUP_SPEC)
    label_map, class_names = build_label_mapper(groups)
    print("类别顺序：", class_names)

    # 提取 X/y（仅保留定义的四类故障样本）
    X, y = extract_X_y(results, feature_indices, label_map)
    print(f"样本数: {len(y)}, 特征维度: {X.shape[1]}")

    # 训练/测试划分（保存测试集在“过滤后序列”中的索引）
    n_all = len(y)
    idx_all = np.arange(n_all)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    n_classes = len(class_names)

    # ================== 使用 GMM + 标签后验映射 的无监督方法 ==================
    print(f"使用 GMM + 标签后验映射的无监督方法进行建模（n_classes={n_classes}）...")
    # 你可以根据需要，把 n_components=2*n_classes 等改成更多成分
    y_prob, y_pred, gmm, comp_fault_prob = fit_gmm_and_get_probabilities(
        X_tr=X_tr,
        y_tr=y_tr,
        X_te=X_te,
        n_classes=n_classes,
        random_state=RANDOM_STATE,
        n_components=5 * n_classes  # 或者 2*n_classes、3*n_classes 等
    )

    # 评估（仍然使用监督标签 y_te 进行评估）
    print("在测试集上评估（基于 GMM 的概率映射结果）...")
    acc = accuracy_score(y_te, y_pred)
    print(f"准确率: {acc:.4f}")
    print("分类报告:")
    print(classification_report(y_te, y_pred, target_names=class_names, digits=4))
    print("混淆矩阵(行=真实, 列=预测):")
    print(confusion_matrix(y_te, y_pred))

    # ========= 概率曲线（在“某一故障条件”的测试样本上绘制四类概率） =========
    # 需要把 idx_te（在过滤后序列中的索引）映射回原始 results 的行索引。
    detailed_labels_full = results[:, INDEX["label"]].astype(np.int32)
    keep_mask = np.array([det in label_map for det in detailed_labels_full], dtype=bool)
    X_all_full = results[:, feature_indices].astype(np.float64)
    mask_ok = np.isfinite(X_all_full).all(axis=1)
    final_mask = keep_mask & mask_ok
    filtered_to_full_indices = np.where(final_mask)[0]  # 长度 = len(y)
    # 将测试子集索引映射为原始 results 的索引
    test_indices_in_full = filtered_to_full_indices[idx_te]

    # ========= 概率曲线：四种真实故障条件，分别画四张单独的图 =========
    if SAVE_PROB_FIG:
        # 若要保存，则用 PROB_FIG_OUT 的目录作为保存目录
        save_dir = os.path.dirname(PROB_FIG_OUT) or "fig_out"
    else:
        save_dir = None  # 直接 plt.show() 四张图

    plot_fault_probabilities_for_all_faults_separate(
        results=results,
        class_names=class_names,
        y_prob=y_prob,
        test_indices_in_full=test_indices_in_full,
        save_dir=save_dir
    )
    # ========= t-SNE 可视化（测试集，按诊断结果上色） =========

    plot_tsne_of_test_samples(
        X_te=X_te,
        y_pred=y_pred,
        save_path=None  # 若想直接显示不保存，传 None
    )