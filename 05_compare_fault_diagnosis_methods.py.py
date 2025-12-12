# -*- coding: utf-8 -*-
"""
F02_E10_method_compare.py

目的：
- 不改动 F02_E09_figure9.py 的主代码，只是“复用其数据加载和预处理方式”，
  在一个独立脚本中对比 6 种方法在“四类故障诊断（阶段 2）”上的效果。

对比方法（共 6 种）：
1) GMM_posterior     : GMM + 标签后验映射（无监督，参考 F02_E09_figure9.py 的思路）
2) Sup_LR            : 监督多类 Logistic Regression
3) Sup_SVM           : 监督 RBF 核 SVM
4) KMeans_posterior  : KMeans + 标签后验映射（无监督）
5) Agglo_posterior   : AgglomerativeClustering + 标签后验映射（无监督）
6) Spectral_posterior: SpectralClustering + 标签后验映射（无监督）

输出：
- 每种方法的 accuracy & classification_report
- 每种方法的混淆矩阵图（4×4，行=真实，列=预测）
- 汇总各方法的 [Accuracy, Macro-Precision, Macro-Recall, Macro-F1] 柱状图对比

注意：
- 直接复用 F02_E09_figure9.py 的：
  - MAT 文件路径 / 布局
  - 特征选择方式（DEFAULT_FEATURES + parse_features）
  - 故障分组方式（DEFAULT_GROUP_SPEC + parse_group_spec + build_label_mapper）
  - 提取 X, y 的方式（extract_X_y）
- 仅在本脚本中重新做一次 train_test_split，不调用 F02_E09_figure9.py 的主流程。
"""

import os
import warnings
from typing import Tuple, Dict, Callable

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ===== 从 F02_E09_figure9.py 导入：数据加载与预处理方式 =====
# 请确保 F02_E09_figure9.py 与本脚本在同一目录（或在 sys.path 中）
from F02_E09_figure9 import (
    MAT_PATH,
    DEFAULT_GROUP_SPEC,
    DEFAULT_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    load_comprehensive_results,
    parse_features,
    parse_group_spec,
    build_label_mapper,
    extract_X_y,
    setup_chinese_font,
)

# ---------------- 全局绘图风格（为了论文子图更清晰） ----------------
_ = setup_chinese_font()  # 确保中文显示正常

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,       # 坐标轴框线粗细
    "xtick.major.width": 1.0,    # x 轴主刻度线宽
    "ytick.major.width": 1.0,    # y 轴主刻度线宽
    "xtick.labelsize": 10,       # x 轴刻度字体
    "ytick.labelsize": 10,       # y 轴刻度字体
    "axes.labelsize": 11,        # 坐标轴标签字体
    "axes.titlesize": 11,        # 标题字体
    "legend.fontsize": 10,
    # 统一使用 Arial 字体
    "font.family": "Arial",
})

# 这里为了画图清晰，用英文类名；与 F02_E09 中的中文对应关系如下：
# "水淹"   -> "Flooding"
# "氧饥饿" -> "Oxygen starvation"
# "膜干"   -> "Membrane drying"
# "氢饥饿" -> "Hydrogen starvation"
CLASS_NAMES_EN = ["Flooding", "Oxygen starvation", "Membrane drying", "Hydrogen starvation"]
N_CLASSES = 4  # 四类故障

# 混淆矩阵中显示的简短标签
CLASS_SHORT = ["F1", "F2", "F3", "F4"]


# ================ 通用工具函数 ================

def plot_confusion_matrix_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names,
    title: str = "",
    cmap=plt.cm.Blues,
):
    """
    绘制带数字和百分比标注的多类混淆矩阵。
    行：真实标签
    列：预测标签
    """
    cm = confusion_matrix(y_true, y_pred)

    # 图像尺寸稍大一点（论文子图常见尺寸，如 3.0 × 2.6 英寸）
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)

    # 使用简短标签 F1, F2, F3, F4
    ax.set_xticks(np.arange(len(CLASS_SHORT)))
    ax.set_yticks(np.arange(len(CLASS_SHORT)))
    ax.set_xticklabels(CLASS_SHORT, rotation=0, ha="center", fontsize=10)
    ax.set_yticklabels(CLASS_SHORT, fontsize=10)
    ax.set_title(title, fontsize=11)

    # 坐标轴四周边框加粗（再额外手动设置一次，确保生效）
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 每一行的总数，用来算百分比
    row_sums = cm.sum(axis=1, keepdims=True)  # [n_classes, 1]

    # 中心点位置微调，让上下两行文字不重叠
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        count_ij = cm[i, j]
        color = "white" if count_ij > thresh else "black"

        total_i = row_sums[i, 0]
        if total_i > 0:
            pct = 100.0 * count_ij / total_i
        else:
            pct = 0.0

        # 上面一行：整数
        ax.text(
            j,
            i - 0.12,
            f"{int(count_ij)}",
            ha="center",
            va="center",
            color=color,
            fontsize=8.5,
        )
        # 下面一行：百分比
        ax.text(
            j,
            i + 0.20,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            color=color,
            fontsize=7.5,
        )

    plt.tight_layout(pad=0.3)
    plt.show()


def compute_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算 Accuracy、macro precision/recall/F1。
    返回：dict
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }


# ================ 数据加载（复用 F02_E09_figure9.py 方式） ================

def load_data_for_fault_4class() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    复用 F02_E09_figure9.py 的方式，从 MAT 文件中读取数据，并提取“四类故障”的 X, y。
    """
    print(f"[数据加载] 从 MAT 文件读取: {MAT_PATH}")
    results = load_comprehensive_results(MAT_PATH)

    # 特征列索引（与 F02_E09 一致）
    feature_indices = parse_features(DEFAULT_FEATURES)
    print(f"[数据加载] 使用特征列: {feature_indices}")

    # 故障分组（与 F02_E09 一致）
    groups = parse_group_spec(DEFAULT_GROUP_SPEC)
    label_map, class_names_cn = build_label_mapper(groups)
    print("[数据加载] 合并后的类别（中文顺序）:", class_names_cn)

    if len(class_names_cn) != 4:
        warnings.warn(
            f"当前分组得到的类别数为 {len(class_names_cn)}，而本脚本假定为 4 类，请检查 DEFAULT_GROUP_SPEC。"
        )

    # 提取 X, y（只保留这几类故障样本）
    X, y = extract_X_y(results, feature_indices, label_map)
    print(f"[数据加载] 样本数: {len(y)}, 特征维度: {X.shape[1]}")
    print(f"[数据加载] y 标签唯一值: {np.unique(y)}")

    return X.astype(np.float64), y.astype(np.int32), class_names_cn


# ================ 各方法实现 ================

# ------ 1) GMM + 标签后验映射（无监督） ------

def fit_gmm_and_get_predictions(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    random_state: int = 42,
    n_components_factor: int = 5,
) -> np.ndarray:
    """
    GMM + 标签后验映射（与 F02_E09_figure9.py 中的核心思想一致）
    """
    n_components = n_components_factor * n_classes

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    gmm.fit(X_tr)

    # 训练集后验：P(component | x_i)
    resp_tr = gmm.predict_proba(X_tr)  # [n_tr, n_components]

    # 统计 P(class | component)
    comp_class_prob = np.zeros((n_components, n_classes), dtype=float)
    for k in range(n_classes):
        mask_k = (y_tr == k).astype(float)  # [n_tr]
        comp_class_prob[:, k] = resp_tr.T @ mask_k  # [n_components]

    # 每个成分的总权重
    comp_sum = comp_class_prob.sum(axis=1, keepdims=True)  # [n_components, 1]
    mask_valid = (comp_sum[:, 0] > 0)  # [n_components]

    # 对有效成分按行归一化
    comp_class_prob[mask_valid, :] /= comp_sum[mask_valid, :]

    # 对无效成分设为均匀分布
    comp_class_prob[~mask_valid, :] = 1.0 / n_classes

    # 测试集后验 P(component | x)
    resp_te = gmm.predict_proba(X_te)  # [n_te, n_components]

    # P(class | x) = sum_j P(class | component=j) * P(component=j | x)
    y_prob = resp_te @ comp_class_prob  # [n_te, n_classes]

    # 数值稳定处理
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    y_pred = y_prob.argmax(axis=1)
    return y_pred


# ------ 2) 监督 Logistic Regression ------

def run_supervised_lr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    """
    多类 Logistic Regression（监督）。
    """
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_pred


# ------ 3) 监督 SVM (RBF) ------

def run_supervised_svm_rbf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    """
    RBF 核 SVM 多类（监督）。
    （保持你原来的 kernel='linear', C=0.05 不变）
    """
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="linear",
                    C=0.05,
                    gamma="scale",
                    probability=False,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_pred


# ------ 4) KMeans + 标签后验映射（无监督） ------

def fit_kmeans_posterior(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    random_state: int = 42,
    n_clusters: int = None,
) -> np.ndarray:
    """
    KMeans + 标签后验映射，无监督聚类方法之一。
    """
    if n_clusters is None:
        n_clusters = n_classes

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    kmeans.fit(X_tr)

    labels_tr = kmeans.labels_  # [n_tr]
    comp_class_prob = np.zeros((n_clusters, n_classes), dtype=float)

    # 统计 P(class | cluster)
    for c in range(n_clusters):
        mask_c = labels_tr == c
        if not mask_c.any():
            comp_class_prob[c, :] = 1.0 / n_classes
            continue
        for k in range(n_classes):
            comp_class_prob[c, k] = np.sum(y_tr[mask_c] == k)
        s = comp_class_prob[c, :].sum()
        if s > 0:
            comp_class_prob[c, :] /= s
        else:
            comp_class_prob[c, :] = 1.0 / n_classes

    # 测试集 -> 找最近簇 -> 使用簇的类别分布作为 P(class | x)
    dist_te = pairwise_distances(X_te, kmeans.cluster_centers_)  # [n_te, n_clusters]
    cluster_idx = np.argmin(dist_te, axis=1)  # [n_te]

    y_prob = np.zeros((X_te.shape[0], n_classes), dtype=float)
    for i, c in enumerate(cluster_idx):
        y_prob[i, :] = comp_class_prob[c, :]

    y_pred = y_prob.argmax(axis=1)
    return y_pred


# ------ 5) 层次聚类 + 最近簇中心 + 后验映射（无监督） ------

def fit_agglomerative_posterior(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    n_clusters: int = None,
) -> np.ndarray:
    """
    AgglomerativeClustering + 标签后验映射。
    为了能在测试集上分配簇，引入“聚类中心”(训练后每簇样本均值)。
    """
    if n_clusters is None:
        n_clusters = n_classes

    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )
    labels_tr = agg.fit_predict(X_tr)  # [n_tr]

    # 计算每个簇的"中心"（均值）
    centers = np.zeros((n_clusters, X_tr.shape[1]), dtype=float)
    for c in range(n_clusters):
        mask_c = labels_tr == c
        if not mask_c.any():
            centers[c, :] = 0.0
        else:
            centers[c, :] = X_tr[mask_c].mean(axis=0)

    # 统计 P(class | cluster)
    comp_class_prob = np.zeros((n_clusters, n_classes), dtype=float)
    for c in range(n_clusters):
        mask_c = labels_tr == c
        if not mask_c.any():
            comp_class_prob[c, :] = 1.0 / n_classes
            continue
        for k in range(n_classes):
            comp_class_prob[c, k] = np.sum(y_tr[mask_c] == k)
        s = comp_class_prob[c, :].sum()
        if s > 0:
            comp_class_prob[c, :] /= s
        else:
            comp_class_prob[c, :] = 1.0 / n_classes

    # 测试集：按最近 center 分配簇
    dist_te = pairwise_distances(X_te, centers)
    cluster_idx = np.argmin(dist_te, axis=1)  # [n_te]

    y_prob = np.zeros((X_te.shape[0], n_classes), dtype=float)
    for i, c in enumerate(cluster_idx):
        y_prob[i, :] = comp_class_prob[c, :]
    y_pred = y_prob.argmax(axis=1)
    return y_pred


# ------ 6) 谱聚类 + 后验映射（无监督） ------

def fit_spectral_posterior(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    random_state: int = 42,
    n_clusters: int = None,
) -> np.ndarray:
    """
    SpectralClustering + 标签后验映射。
    同样使用“每簇样本均值作为中心”以便在测试集上分配簇。
    注意：谱聚类在样本很多时可能较慢，请按你数据规模决定是否使用。
    """
    from sklearn.cluster import SpectralClustering  # 局部导入，避免环境无此模块时报错过早

    if n_clusters is None:
        n_clusters = n_classes

    spec = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=random_state,
        n_neighbors=10,
        n_init=10,
        assign_labels="kmeans",
    )
    labels_tr = spec.fit_predict(X_tr)  # [n_tr]

    centers = np.zeros((n_clusters, X_tr.shape[1]), dtype=float)
    for c in range(n_clusters):
        mask_c = labels_tr == c
        if not mask_c.any():
            centers[c, :] = 0.0
        else:
            centers[c, :] = X_tr[mask_c].mean(axis=0)

    comp_class_prob = np.zeros((n_clusters, n_classes), dtype=float)
    for c in range(n_clusters):
        mask_c = labels_tr == c
        if not mask_c.any():
            comp_class_prob[c, :] = 1.0 / n_classes
            continue
        for k in range(n_classes):
            comp_class_prob[c, k] = np.sum(y_tr[mask_c] == k)
        s = comp_class_prob[c, :].sum()
        if s > 0:
            comp_class_prob[c, :] /= s
        else:
            comp_class_prob[c, :] = 1.0 / n_classes

    dist_te = pairwise_distances(X_te, centers)
    cluster_idx = np.argmin(dist_te, axis=1)

    y_prob = np.zeros((X_te.shape[0], n_classes), dtype=float)
    for i, c in enumerate(cluster_idx):
        y_prob[i, :] = comp_class_prob[c, :]
    y_pred = y_prob.argmax(axis=1)
    return y_pred


# ================ 指标柱状图绘制 ================

def plot_metrics_bar_chart(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names=("accuracy", "macro_precision", "macro_recall", "macro_f1"),
):
    """
    metrics_dict: {method_name: {"accuracy":..., "macro_precision":...,...}}

    调整：
    - 图像略宽一些，便于作为横放的论文子图；
    - 柱子边框加粗，字体略放大；
    - 为每种方法指定固定颜色，其中 GMM 为红色，Agglo 远离红色。
    """
    methods = list(metrics_dict.keys())
    n_methods = len(methods)
    n_metrics = len(metric_names)

    # 为各方法指定颜色（可根据需要微调）
    method_colors = {
        "GMM":      "#e41a1c",  # 亮红色（重点方法）
        "Sup_LR":   "#377eb8",  # 蓝
        "Sup_SVM":  "#ff7f00",  # 橙
        "KMeans":   "#4daf4a",  # 绿
        "Agglo":    "#4d4d4d",  # 深灰，与红色有明显区别
        "Spectral": "#984ea3",  # 紫
    }

    # 构建一个 [n_methods, n_metrics] 的数组
    values = np.zeros((n_methods, n_metrics), dtype=float)
    for mi, m in enumerate(methods):
        for ki, metric in enumerate(metric_names):
            values[mi, ki] = metrics_dict[m][metric]

    # 计算全局最小和最大，用于设定 y 轴范围
    v_min = values.min()
    v_max = values.max()
    margin = (v_max - v_min) * 0.2 if v_max > v_min else 0.02
    y_min = max(0.0, v_min - margin)
    if y_min < 0.7:
        y_min = 0.7
    y_max = 1.05

    x = np.arange(n_metrics)  # 4 个组
    total_width = 0.8
    bar_width = total_width / n_methods

    plt.figure(figsize=(6, 2.3), dpi=300)

    for mi, m in enumerate(methods):
        offsets = x - total_width / 2 + (mi + 0.5) * bar_width
        color = method_colors.get(m, None)  # 找不到就用默认颜色

        bars = plt.bar(
            offsets,
            values[mi, :],
            width=bar_width,
            label=m,
            alpha=0.6,
            edgecolor="black",
            linewidth=1.0,
            color=color,  # 使用指定颜色
        )

        # 在每根柱子上标数字
        for xi, v in zip(offsets, values[mi, :]):
            plt.text(
                xi,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    plt.xticks(
        x,
        ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"],
        fontsize=10,
    )
    plt.ylabel("Score", fontsize=11)
    plt.ylim(y_min, y_max)
    plt.title("Comparison of methods on four metrics", fontsize=11)

    plt.legend(
        fontsize=8,
        ncol=min(6, len(methods)),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
    )

    plt.tight_layout(pad=0.4)
    plt.show()


# ================ 主流程 ================

def main():
    warnings.filterwarnings("ignore")

    # 1. 使用 F02_E09_figure9.py 的方式加载“四类故障”数据
    X, y, class_names_cn = load_data_for_fault_4class()

    # 若中文类名数为 4，则在输出文本里既打印中文，又在图中用英文 CLASS_NAMES_EN
    if len(class_names_cn) == 4:
        print("[信息] 中文类别名：", class_names_cn)
    else:
        print("[警告] 类别数 != 4，当前 class_names_cn =", class_names_cn)

    # 2. 训练/测试划分（比例与 F02_E09 一致，使用 TEST_SIZE, RANDOM_STATE）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[数据划分] Train: {X_tr.shape}, Test: {X_te.shape}")

    # 3. 定义 6 种方法及其调用函数
    method_funcs: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
        "GMM": lambda Xtr, ytr, Xte: fit_gmm_and_get_predictions(
            Xtr,
            ytr,
            Xte,
            n_classes=N_CLASSES,
            random_state=RANDOM_STATE,
            n_components_factor=5,
        ),
        "Sup_LR": run_supervised_lr,
        "Sup_SVM": run_supervised_svm_rbf,
        "KMeans": lambda Xtr, ytr, Xte: fit_kmeans_posterior(
            Xtr,
            ytr,
            Xte,
            n_classes=N_CLASSES,
            random_state=RANDOM_STATE,
            n_clusters=5 * N_CLASSES,
        ),
        "Agglo": lambda Xtr, ytr, Xte: fit_agglomerative_posterior(
            Xtr,
            ytr,
            Xte,
            n_classes=N_CLASSES,
            n_clusters=4 * N_CLASSES,
        ),
        "Spectral": lambda Xtr, ytr, Xte: fit_spectral_posterior(
            Xtr,
            ytr,
            Xte,
            n_classes=N_CLASSES,
            random_state=RANDOM_STATE,
            n_clusters=4 * N_CLASSES,
        ),
    }

    # 4. 逐方法训练 & 评估
    all_metrics: Dict[str, Dict[str, float]] = {}

    for method_name, func in method_funcs.items():
        print("\n" + "=" * 80)
        print(f"Method: {method_name}")
        print("=" * 80)

        y_pred = func(X_tr, y_tr, X_te)

        # 文本指标
        acc = accuracy_score(y_te, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report (targets in order of CLASS_NAMES_EN):")
        print(classification_report(y_te, y_pred, target_names=CLASS_NAMES_EN, digits=4))
        print("Confusion matrix (row = true, col = pred):")
        print(confusion_matrix(y_te, y_pred))

        # 混淆矩阵图
        plot_confusion_matrix_cm(
            y_true=y_te,
            y_pred=y_pred,
            class_names=CLASS_NAMES_EN,
            title=f"Confusion Matrix - {method_name}",
        )

        # 存储宏指标
        all_metrics[method_name] = compute_macro_metrics(y_te, y_pred)

    # 5. 绘制各方法指标柱状对比图
    print("\nSummary metrics (per method):")
    for m, md in all_metrics.items():
        print(f"{m}: " + ", ".join([f"{k}={v:.4f}" for k, v in md.items()]))

    plot_metrics_bar_chart(all_metrics)


if __name__ == "__main__":
    main()