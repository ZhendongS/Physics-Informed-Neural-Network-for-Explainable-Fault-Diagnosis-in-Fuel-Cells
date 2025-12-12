# -*- coding: utf-8 -*-
"""
D07_fault_diagnosis_figure6＆7.py（增强版）
需求改造：
1) 增加可视化：若选择的“散点绘图特征”维数>2，则自动用 t-SNE 将其压到2D再绘制散点；
   在代码最前边通过 PLOT_FEATURES 选择要用于散点可视化的特征（名称或列号）。
2) 顶部定义四组分类特征 FEAT_GRP1..FEAT_GRP4；通过 FEATURES_TO_RUN 选择要跑的组。
   对每一组特征：给出分类结果，并绘制ROC曲线（统一绘制在一张图中）。
3) 保持原有绘图风格与字号等设置不变或尽量一致。
"""

import os
import re
import argparse
import warnings
import numpy as np
import scipy.io as sio
from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import IsolationForest
# 可选的降维
from sklearn.manifold import TSNE

# 作图
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager


# ======================== 顶部参数（按需修改） ========================

# 数据文件
DEFAULT_MAT_PATH = "F01_output.mat"

# 四组分类特征（支持名称或列号，英文逗号分隔）
FEAT_GRP1 = "epi,res"
FEAT_GRP2 = "x0,x3,x4,x5"
FEAT_GRP3 = "res"
FEAT_GRP4 = "y_true"

# 要运行的特征组编号
FEATURES_TO_RUN = [1, 2, 3, 4]

# 用于散点可视化的特征（若>2维，自动用t-SNE降到2D）
# 保持原风格：若选择正好2个特征，则直接作为X/Y轴；否则t-SNE
PLOT_FEATURES = "x0,x3,x4,x5"  # 可改成 "res,epi"（两维直接散点）

# 标签合并（顺序即类别顺序；新布局：第17列为细分标签ID）
DEFAULT_GROUP_SPEC = "正常:0 | 水淹:1,2,3 | 氧饥饿:4,5,6 | 膜干:7,8,9 | 氢饥饿:10,11,12"
DEFAULT_GROUP_SPEC = "正常:0 | 故障:1,2,3,4,5,6,7,8,9 ,10,11,12"
# 数据划分/模型
DEFAULT_TEST_SIZE = 0.9
DEFAULT_RANDOM_STATE = 49
DEFAULT_BALANCED = True
DEFAULT_SHOW_COEF = 5  # 每类显示前N个正/负重要特征（0为不显示）

# t-SNE 参数（仅用于可视化）
TSNE_PARAMS = dict(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=DEFAULT_RANDOM_STATE)

# ======================== 字体与索引 ========================

def setup_chinese_font():
    """设置matplotlib的中文字体支持（保持原风格）"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体设置成功")
        return True
    except Exception:
        try:
            font_path = font_manager.findfont(font_manager.FontProperties(family=['sans-serif']))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            print("使用默认字体")
            return False
        except Exception:
            print("字体设置失败，使用英文标签")
            return False

chinese_support = setup_chinese_font()

# 新布局索引
INDEX = {
    **{f"x{i}": i for i in range(8)},  # x0..x7 -> 0..7
    "y_true": 8,
    "y_pred": 9,
    "ale": 10,
    "epi": 11,
    "res": 12,
    "pV": 13,
    "pT": 14,
    "pH": 15,
    "pO": 16,
    "label": 17
}
REQUIRED_MAX_INDEX = max(INDEX.values())  # 17

# ======================== 工具函数 ========================

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
        if re.match(r"^-?\d+$", t):
            idx = int(t)
        else:
            if t not in INDEX:
                raise KeyError(f"未知特征名称: '{t}'。可用名称: {list_available_features()}")
            idx = INDEX[t]
        if idx == INDEX["label"]:
            raise ValueError("不允许将 'label' 作为输入特征。")
        indices.append(idx)
    # 去重保持顺序
    seen = set()
    ordered = []
    for idx in indices:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    if INDEX["y_true"] in ordered:
        warnings.warn("特征中包含 y_true（真实输出），可能导致目标泄露。仅在可解释对比时使用。")
    return ordered

def parse_group_spec(spec: str) -> Dict[str, List[int]]:
    parts = re.split(r"[|；;]\s*|\n+", spec.strip())
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
        raise ValueError("未解析到任何分组，请检查 GROUP_SPEC。")
    return groups

def build_label_mapper(groups: Dict[str, List[int]]) -> Tuple[Dict[int, int], List[str]]:
    class_names = list(groups.keys())
    detail_to_coarse: Dict[int, int] = {}
    for coarse_idx, name in enumerate(class_names):
        for det in groups[name]:
            if det in detail_to_coarse:
                prev = class_names[detail_to_coarse[det]]
                raise ValueError(f"细分标签 {det} 被多个组包含：'{prev}' 与 '{name}'")
            detail_to_coarse[det] = coarse_idx
    return detail_to_coarse, class_names

def extract_X_y(results: np.ndarray, feature_indices: List[int], label_map: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    detailed_labels = results[:, INDEX["label"]].astype(np.int32)
    keep_mask = np.array([det in label_map for det in detailed_labels], dtype=bool)
    X_all = results[keep_mask][:, feature_indices].astype(np.float64)
    y_all = np.array([label_map[int(d)] for d in detailed_labels[keep_mask]], dtype=np.int32)
    finite_mask = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
    return X_all[finite_mask], y_all[finite_mask]

def build_classifier(balanced: bool = False) -> Pipeline:
    class_weight = "balanced" if balanced else None
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=DEFAULT_RANDOM_STATE,
            class_weight=class_weight
        ))
    ])
    return clf

def explain_coefficients(clf: Pipeline, feature_indices: List[int], class_names: List[str], topn: int = 5):
    if topn <= 0:
        return
    inv_map = {v: k for k, v in INDEX.items()}
    feat_names = [inv_map.get(i, f"col{i}") for i in feature_indices]
    try:
        lr = clf.named_steps["logreg"]
        coefs = lr.coef_
    except Exception as e:
        print(f"无法提取系数: {e}")
        return
    print("\n每个类别的特征重要性（基于LR系数，标准化空间）：")
    for c_idx, cname in enumerate(class_names):
        w = coefs[c_idx]
        order_pos = np.argsort(-w)[:topn]
        order_neg = np.argsort(w)[:topn]
        pos_list = [f"{feat_names[i]}(+{w[i]:.3f})" for i in order_pos]
        neg_list = [f"{feat_names[i]}({w[i]:.3f})" for i in order_neg]
        print(f"- 类别[{c_idx}] {cname}:")
        print(f"  正向贡献TOP{topn}: " + ", ".join(pos_list))
        print(f"  负向贡献TOP{topn}: " + ", ".join(neg_list))

# ======================== 可视化（保持原风格） ========================

def plot_two_scatter_views(results: np.ndarray, label_col: int = 17):
    """
    原有论文友好版双散点图：
      1) 温度(出口 x5) vs 电压(y_true)
      2) 残差res vs 认知不确定度epi
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [ "Arial","SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial"],
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "axes.linewidth": 2.0,
        "grid.alpha": 0.5,
    })
    idx = {
        "x0": 0, "x1": 1, "x2": 2, "x3": 3, "x4": 4, "x5": 5, "x6": 6, "x7": 7,
        "y_true": 8, "y_pred": 9, "ale": 10, "epi": 11, "res": 12, "pV": 13, "pT": 14, "pH": 15, "pO": 16,
        "label": label_col
    }
    temp_col = idx["x5"]
    voltage_col = idx["y_true"]
    res_col = idx["res"]
    epi_col = idx["epi"]
    temp = results[:, temp_col].astype(float).squeeze()
    voltage = results[:, voltage_col].astype(float).squeeze()
    res = results[:, res_col].astype(float).squeeze()
    epi = results[:, epi_col].astype(float).squeeze()
    labels = results[:, idx["label"]].astype(int).squeeze()
    mask_finite_1 = np.isfinite(temp) & np.isfinite(voltage) & np.isfinite(labels)
    mask_finite_2 = np.isfinite(res) & np.isfinite(epi) & np.isfinite(labels)
    temp, voltage, labels1 = temp[mask_finite_1], voltage[mask_finite_1], labels[mask_finite_1]
    res, epi, labels2 = res[mask_finite_2], epi[mask_finite_2], labels[mask_finite_2]
    normal_mask_1 = (labels1 == 0); fault_mask_1 = ~normal_mask_1
    normal_mask_2 = (labels2 == 0); fault_mask_2 = ~normal_mask_2
    color_normal = "#1f77b4"; color_fault = "#d62728"
    s_point = 18; alpha_point = 0.6

    plt.figure(figsize=(7.2, 6.0), dpi=300)
    plt.scatter(temp[fault_mask_1],  voltage[fault_mask_1],  s=s_point, c=color_fault,  alpha=alpha_point, label="Fault data point", edgecolors='none')
    plt.scatter(temp[normal_mask_1], voltage[normal_mask_1], s=s_point, c=color_normal, alpha=alpha_point, label="Normal data point", edgecolors='none')
    plt.xlabel("Stack temperature (°C)"); plt.ylabel("Stack voltage (V)")
    plt.grid(True, ls=":", alpha=0.4); plt.legend(frameon=False); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7.2, 6.0), dpi=300)
    plt.scatter(res[fault_mask_2],   epi[fault_mask_2],   s=s_point, c=color_fault,  alpha=alpha_point, label="Fault data point", edgecolors='none')
    plt.scatter(res[normal_mask_2],  epi[normal_mask_2],  s=s_point, c=color_normal, alpha=alpha_point, label="Normal data point", edgecolors='none')
    plt.xlabel("Model residual (V)"); plt.ylabel("Epistemic uncertainty (V)")
    plt.grid(True, ls=":", alpha=0.4); plt.legend(frameon=False); plt.tight_layout(); plt.show()

def plot_scatter_by_features(results: np.ndarray, feature_indices: List[int],
                             label_map: Dict[int, int], class_names: List[str]):
    """
    新增：按 PLOT_FEATURES 选择的特征绘制散点。
    - 若恰好2维：直接用这两维做X/Y轴；
    - 若>2维：使用t-SNE降到2D再画散点。
    绘图风格尽量与 plot_two_scatter_views 保持一致：
      - 点大小、透明度
      - 颜色方案
      - 字体、坐标轴样式
      - 图例样式（frameon=False）
    说明：
      - 这里仍然区分多类别，每个 coarse 类一个颜色；
      - 若只有“正常/故障”两类，将与原函数配色最为接近。
    """

    # ===== 数据准备（只保留映射后的样本）=====
    detailed_labels = results[:, INDEX["label"]].astype(np.int32)
    keep_mask = np.array([det in label_map for det in detailed_labels], dtype=bool)
    X = results[keep_mask][:, feature_indices].astype(np.float64)
    y_det = detailed_labels[keep_mask]
    y = np.array([label_map[int(d)] for d in y_det], dtype=np.int32)

    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]
    if X.shape[0] == 0:
        print("plot_scatter_by_features：无有效样本，跳过绘图。")
        return

    # ===== 维数处理：2 维直接画，多于 2 维走 t-SNE =====
    if X.shape[1] > 2:
        print(f"散点可视化：特征维度={X.shape[1]}，使用 t-SNE 映射到2D。")
        X_2d = TSNE(**TSNE_PARAMS).fit_transform(X)
        x_plot, y_plot = X_2d[:, 0], X_2d[:, 1]
        xlabel = "t-SNE dim1"
        ylabel = "t-SNE dim2"
    else:
        x_plot, y_plot = X[:, 0], X[:, 1]
        inv_map = {v: k for k, v in INDEX.items()}
        xlabel = inv_map.get(feature_indices[0], f"col{feature_indices[0]}")
        ylabel = inv_map.get(feature_indices[1], f"col{feature_indices[1]}")

    # ===== 字体 / 全局风格：与 plot_two_scatter_views 对齐 =====
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial","SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "axes.linewidth": 2.0,
        "grid.alpha": 0.5,
    })

    # ===== 配色和点样式：仿照 plot_two_scatter_views =====
    s_point = 18
    alpha_point = 0.6
    color_normal = "#1f77b4"  # 蓝色
    color_fault = "#d62728"   # 红色

    # coarse 类个数
    n_classes = len(class_names)

    # 如果正好是两类，并且类名中包含“正常”这一类，则尽量使用“正常-蓝 / 故障-红”的风格
    use_binary_normal_fault_style = False
    try:
        idx_normal = class_names.index("正常")
        if n_classes == 2:
            use_binary_normal_fault_style = True
            idx_fault = 1 - idx_normal
    except ValueError:
        idx_normal = None

    plt.figure(figsize=(7.2, 6.0), dpi=300)

    if use_binary_normal_fault_style:
        # 二类情况：保持与 plot_two_scatter_views 同风格
        mask_normal = (y == idx_normal)
        mask_fault = ~mask_normal
        plt.scatter(
            x_plot[mask_fault],
            y_plot[mask_fault],
            s=s_point,
            c=color_fault,
            alpha=alpha_point,
            label="Fault data point",
            edgecolors="none",
        )
        plt.scatter(
            x_plot[mask_normal],
            y_plot[mask_normal],
            s=s_point,
            c=color_normal,
            alpha=alpha_point,
            label="Normal data point",
            edgecolors="none",
        )
    else:
        # 多类情况：正常类若存在用蓝，其余类用 tab10 做区分
        cmap = plt.cm.get_cmap("tab10", max(n_classes, 2))

        for cls in range(n_classes):
            mask = (y == cls)
            if idx_normal is not None and cls == idx_normal:
                c = color_normal
                label = "Normal data point"
            else:
                # 故障类：用红色 + 其他颜色区分
                if cls == 0 and idx_normal is None:
                    # 没有显式“正常”，第0类采用蓝色
                    c = color_normal
                else:
                    c = cmap(cls % cmap.N)
                label = class_names[cls]
            plt.scatter(
                x_plot[mask],
                y_plot[mask],
                s=s_point,
                alpha=alpha_point,
                c=[c],
                label=label,
                edgecolors="none",
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, ls=":", alpha=0.4)

    # 图例：左上角 + 浅色半透明底色，减少对数据点的遮挡感
    leg = plt.legend(
        loc="upper left",
        fontsize=16,
        frameon=True,  # 开启图例边框/背景
    )

    # 设置图例底色和透明度
    frame = leg.get_frame()
    frame.set_facecolor("white")  # 或者 "#f0f0f0" 之类的浅灰
    frame.set_alpha(0.8)  # 透明度（0~1），数值越小越透明
    frame.set_edgecolor("black")  # 可选：边框颜色
    frame.set_linewidth(0.8)  # 可选：边框线宽

    plt.tight_layout()
    plt.show()

def plot_binary_roc(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1, title: str = "ROC（二分类）"):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    #plt.plot(fpr, tpr, lw=2, label=f"{title} AUC={roc_auc:.3f}")
    plt.plot(fpr, tpr, lw=2, label=f"")

# ======================== 主流程 ========================


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="故障分类（四组特征+可视化+t-SNE）")
    parser.add_argument("--mat", type=str, default=DEFAULT_MAT_PATH, help="D06_output.mat 路径")
    parser.add_argument("--group-spec", type=str, default=DEFAULT_GROUP_SPEC, help="标签组合定义")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="测试集比例")
    parser.add_argument("--balanced", action="store_true", default=DEFAULT_BALANCED, help="类别平衡")
    parser.add_argument("--show-coef", type=int, default=DEFAULT_SHOW_COEF, help="每类显示的正/负重要特征数（0关闭）")
    parser.add_argument("--list-features", action="store_true", help="列出可用特征名称并退出")
    args = parser.parse_args()

    if args.list_features:
        print("可用特征名称（新布局）：")
        for name in list_available_features():
            print(f"  {name} -> 列 {INDEX[name]}")


    # 1) 加载数据与标签映射
    print(f"加载数据: {args.mat}")
    results = load_comprehensive_results(args.mat)
    groups = parse_group_spec(args.group_spec)
    label_map, class_names = build_label_mapper(groups)
    print("类别定义（输出顺序）：")
    for i, name in enumerate(class_names):
        print(f"  类别{i}: {name} -> 细分 {groups[name]}")

    # 2) 散点可视化（按 PLOT_FEATURES，支持t-SNE）
    try:
        plot_feat_idx = parse_features(PLOT_FEATURES)
        if len(plot_feat_idx) < 2:
            print(f"PLOT_FEATURES 至少需要2个特征，当前为 {len(plot_feat_idx)}，跳过散点可视化。")
        else:
            plot_scatter_by_features(results, plot_feat_idx, label_map, class_names)
    except Exception as e:
        print(f"散点可视化失败：{e}")

    # 3) 四组特征评估 + ROC 绘制在一张图
    feat_groups = {
        1: FEAT_GRP1,
        2: FEAT_GRP2,
        3: FEAT_GRP3,
        4: FEAT_GRP4,
    }
    # ROC 颜色配置：第 1 组深蓝，其它组低饱和度
    ROC_COLORS = [
        "#d62728",  # 组1：深蓝（dark blue）
        "#f5b482",  # 组2：低饱和浅蓝灰
        "#acd78e",  # 组3：低饱和浅绿
        "#c1acd5",  # 组4：低饱和浅紫
    ]
    # ROC 画布（保持原风格，单图多曲线）
    plt.figure(figsize=(6.5, 5.5), dpi=300)
    plt.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    # 加粗当前坐标轴的边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)  # 这里改成你想要的粗细，比如 2.0 / 3.0


    for gid in FEATURES_TO_RUN:
        spec = feat_groups.get(gid, None)
        if not spec:
            print(f"[跳过] 未定义特征组 {gid}")
            continue

        try:
            fidx = parse_features(spec)
        except Exception as e:
            print(f"[跳过] 特征组 {gid} 解析错误: {e}")
            continue

        # 提取数据
        X, y = extract_X_y(results, fidx, label_map)
        if len(y) == 0:
            print(f"[跳过] 特征组 {gid} 无有效样本")
            continue

        # 划分
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=DEFAULT_RANDOM_STATE, stratify=y)

        # 训练（LogisticRegression，保持原风格）
        clf = build_classifier(balanced=args.balanced)
        print("\n" + "=" * 60)
        inv_map = {v: k for k, v in INDEX.items()}
        feat_names = [inv_map.get(i, f"col{i}") for i in fidx]
        print(f"特征组 {gid}: {spec} -> {feat_names}")
        print(f"训练分类器（逻辑回归, balanced={args.balanced}）...")
        clf.fit(X_tr, y_tr)

        # 评估
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)
        acc = accuracy_score(y_te, y_pred)
        print(f"准确率: {acc:.4f}")
        print("分类报告:")
        print(classification_report(y_te, y_pred, target_names=class_names, digits=4))
        print("混淆矩阵(行=真实, 列=预测):")
        print(confusion_matrix(y_te, y_pred))

        # 系数解释
        #explain_coefficients(clf, fidx, class_names, topn=args.show_coef)

        # ROC（将多类问题简化为“正常 vs 其余”二分类，与原示范一致）
        # 识别“正常”类别索引：优先找中文名“正常”，否则取第0类
        try:
            normal_idx = class_names.index("正常")
        except ValueError:
            normal_idx = 0
        y_true_bin = (y_te != normal_idx).astype(int)  # 故障=1
        # 概率中取“故障”总体的分数：这里简单地用 1 - P(正常)
        p_normal = y_prob[:, normal_idx]
        p_fault = 1.0 - p_normal
        fpr, tpr, _ = roc_curve(y_true_bin, p_fault, pos_label=1)
        roc_auc = auc(fpr, tpr)
        # 选择对应的颜色：第1组用深蓝，2、3、4组用低饱和色
        base_color = ROC_COLORS[gid - 1] if 1 <= gid <= len(ROC_COLORS) else "#666666"

        plt.plot(
            fpr,
            tpr,
            lw=2,
            color=base_color,
            label=f""  # 需要图例时可加上文字
        )
        # —— 在终端打印有监督方法的 AUC ——
        print(f"[监督] 特征组 {gid}（LogisticRegression）AUC = {roc_auc:.4f}")

        # ====================== 无监督 ROC：仅对第一组特征（FEAT_GRP1 = "epi,res"） ======================
        if gid == 1:
            # 使用 IsolationForest 作为无监督异常检测示例
            # 典型做法：只用“正常”样本训练无监督模型
            mask_tr_normal = (y_tr == normal_idx)
            if np.sum(mask_tr_normal) > 10:  # 至少要有一些正常样本
                X_tr_unsup = X_tr[mask_tr_normal]
            else:
                # 如果正常样本太少，就退化为用全部训练样本
                X_tr_unsup = X_tr
                print("警告：正常样本太少，无监督模型改为在全部训练集上拟合。")

            iso = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=DEFAULT_RANDOM_STATE
            )
            iso.fit(X_tr_unsup)

            # 对测试集给“异常得分”：score_samples 越小越异常，这里取负号使“越大越异常”
            anomaly_score = -iso.score_samples(X_te)

            fpr_u, tpr_u, _ = roc_curve(y_true_bin, anomaly_score, pos_label=1)
            roc_auc_u = auc(fpr_u, tpr_u)
            # —— 在终端打印无监督方法的 AUC ——
            print(f"[无监督] 特征组 {gid}（IsolationForest）AUC = {roc_auc_u:.4f}")
            # 关键点：颜色和有监督那条保持一致，只把线型改为虚线
            # 为此，先从上面那条线取颜色，然后画一条新的虚线
            # （如果你想更显式，也可以直接指定颜色，如 color="C0"）
            last_line = plt.gca().lines[-1]
            supervised_color = last_line.get_color()

            # 与有监督同色，但虚线
            plt.plot(
                fpr_u,
                tpr_u,
                lw=2,
                ls="--",
                color=ROC_COLORS[0],  # 和上面那条有监督 ROC 的 base_color 一致
                label=f""
            )

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 保留原有双散点示例
    plot_two_scatter_views(results, label_col=INDEX["label"])
