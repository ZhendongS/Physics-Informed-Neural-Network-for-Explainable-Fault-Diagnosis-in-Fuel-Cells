# P2指的是本测试中正常数据采用极化测试-2
# 可解释性得到了增强
# 配合E08_figure8,载入本代码的输出文件：F01_output.mat
# 功能数据处理、训练得到模型估计结果和多物理场物理损失

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torch.optim.lr_scheduler import StepLR
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import platform


# 设置中文字体支持
def setup_chinese_font():
    """设置matplotlib的中文字体支持"""
    try:
        # 方法1: 使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("中文字体设置成功")
        return True
    except:
        try:
            # 方法2: 使用matplotlib自带字体
            font_path = font_manager.findfont(font_manager.FontProperties(family=['sans-serif']))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            print("使用默认字体")
            return False
        except:
            print("字体设置失败，使用英文标签")
            return False


# 在代码开头调用
chinese_support = setup_chinese_font()


# ↑ 库函数 ______________________________________________________
def add_noise_to_combined_data(Y_data, noise_type='gaussian', noise_level=0.02,
                               noise_target='fault_only', seed=42):
    """
    在合并数据上添加噪声

    Parameters:
    - noise_target: 'fault_only', 'normal_only', 'all', 'random'
    """
    np.random.seed(seed)

    signal_std = np.std(Y_data)
    noise_std = noise_level * signal_std

    # 生成噪声
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_std, Y_data.shape)
    elif noise_type == 'uniform':
        noise_range = noise_std * np.sqrt(12)
        noise = np.random.uniform(-noise_range / 2, noise_range / 2, Y_data.shape)
    else:
        raise ValueError("Unsupported noise type")

    # 创建噪声掩码
    noise_mask = np.zeros(len(Y_data), dtype=bool)

    if noise_target == 'fault_only':
        # 只在故障数据上添加噪声（假设故障数据在后面）
        # 这需要外部提供正常数据的长度信息
        pass  # 在调用时需要额外处理
    elif noise_target == 'all':
        noise_mask[:] = True
    elif noise_target == 'random':
        random_indices = np.random.choice(len(Y_data), size=len(Y_data) // 2, replace=False)
        noise_mask[random_indices] = True

    # 应用噪声
    Y_noisy = Y_data.copy()
    Y_noisy[noise_mask] += noise[noise_mask]

    noise_info = {
        'noise_type': noise_type,
        'noise_level': noise_level,
        'noise_std': noise_std,
        'noise_mask': noise_mask,
        'affected_samples': np.sum(noise_mask)
    }

    print(f"噪声添加详情:")
    print(f"  噪声类型: {noise_type}")
    print(f"  噪声水平: {noise_level:.3f}")
    print(f"  噪声标准差: {noise_std:.6f}")
    print(f"  受影响样本: {np.sum(noise_mask)}/{len(Y_data)} ({np.sum(noise_mask) / len(Y_data) * 100:.1f}%)")

    return Y_noisy, noise_info


def load_data_normal_raw(data_path):
    """
    加载正常数据，但不进行归一化和数据集划分
    返回原始数据用于后续拼接
    """
    # 加载数据
    data = scipy.io.loadmat(data_path)
    data_length = len(data['I'])

    # 构建输入特征（确保所有数组都是二维的）
    input1_time = np.arange(1, data_length + 1).reshape(-1, 1)  # 转为列向量
    input2_ist = data['I'].reshape(-1, 1) if data['I'].ndim == 1 else data['I']
    input3_Wflow = data['m_W'].reshape(-1, 1) if data['m_W'].ndim == 1 else data['m_W']
    input4_tin = data['T_W_in'].reshape(-1, 1) if data['T_W_in'].ndim == 1 else data['T_W_in']
    input5_PHin = data['P_H_in'].reshape(-1, 1) if data['P_H_in'].ndim == 1 else data['P_H_in']
    input6_POin = data['P_O_in'].reshape(-1, 1) if data['P_O_in'].ndim == 1 else data['P_O_in']
    input7_tout = data['T_W_out'].reshape(-1, 1) if data['T_W_out'].ndim == 1 else data['T_W_out']
    input8_Hflow = data['m_H2'].reshape(-1, 1) if data['m_H2'].ndim == 1 else data['m_H2']
    input9_Aflow = data['m_O2'].reshape(-1, 1) if data['m_O2'].ndim == 1 else data['m_O2']

    # 合并所有输入特征为一个二维数组
    X_data = np.column_stack(
        [input2_ist, input3_Wflow, input4_tin, input5_PHin, input6_POin, input7_tout, input8_Hflow, input9_Aflow])

    # 输出数据
    Y_data = data['U'].reshape(-1, 1) if data['U'].ndim == 1 else data['U']

    # 筛选电流不为0的样本
    valid_indices = np.where((input2_ist > 50) & (input2_ist < 800))[0]
    X_data = X_data[valid_indices]
    Y_data = Y_data[valid_indices]
    # 删除最后的关机数据

    # n_data1 =200
    # n_data2 =2800
    # X_data =X_data[n_data1:n_data2, :]
    # Y_data = Y_data[n_data1:n_data2, :]
    print(f"正常数据加载完成: {X_data.shape[0]} 个样本")

    return X_data, Y_data


def load_data_fault_raw(data_path):
    """
    加载故障数据，但不进行归一化和数据集划分
    返回原始数据用于后续拼接
    """
    # 加载数据
    data_matlab = scipy.io.loadmat(data_path)
    data = data_matlab['segment_double']

    data_length = len(data)

    # 构建输入特征（确保所有数组都是二维的）
    input1_time = np.arange(1, data_length + 1).reshape(-1, 1)  # 转为列向量

    # 合并所有输入特征为一个二维数组
    corr_index = np.array([20, 25, 65, 68, 69, 66, 14, 16]) - 3  # 具体含义需要看onenote
    X_data = data[:, corr_index]

    # 输出数据
    Y_data = data[:, [19 - 3]]

    # 筛选电流不为0的样本
    # 假设电流在第2列（根据实际数据调整）
    input2_ist = X_data[:, 1:2]  # 提取电流列
    valid_indices = np.where(input2_ist != 0)[0]
    X_data = X_data[valid_indices]
    Y_data = Y_data[valid_indices]

    print(f"故障数据加载完成: {X_data.shape[0]} 个样本")

    return X_data, Y_data


def combine_and_normalize_datasets(normal_data, fault_data_list, training_rate=0.8,
                                   noise_config=None, seed=42):
    """
    合并正常数据和故障数据，统一归一化，并划分训练集和测试集
    训练集来自正常数据，测试集包含所有数据
    """
    np.random.seed(seed)

    X_normal, Y_normal = normal_data

    print("=" * 60)
    print("合并数据集并统一处理")
    print("=" * 60)

    # 检查特征维度一致性
    print("检查特征维度一致性:")
    print(f"  正常数据: {X_normal.shape[0]} 样本, {X_normal.shape[1]} 特征")

    # 验证故障数据格式
    if not isinstance(fault_data_list, list):
        raise ValueError("fault_data_list 必须是列表格式")

    for i, fault_item in enumerate(fault_data_list):
        if len(fault_item) != 3:
            raise ValueError(f"故障数据 {i + 1} 格式错误，应为 (X_fault, Y_fault, label)")

        X_fault, Y_fault, label = fault_item
        print(f"  {label}: {X_fault.shape[0]} 样本, {X_fault.shape[1]} 特征")

        if X_fault.shape[1] != X_normal.shape[1]:
            raise ValueError(
                f"{label}的特征数({X_fault.shape[1]})与正常数据({X_normal.shape[1]})不一致!")

    # 合并所有数据用于统一归一化
    all_X_data = [X_normal]
    all_Y_data = [Y_normal]
    data_labels = ['正常数据'] * len(X_normal)

    # 收集所有故障数据
    for X_fault, Y_fault, label in fault_data_list:
        all_X_data.append(X_fault)
        all_Y_data.append(Y_fault)
        data_labels.extend([label] * len(X_fault))

    # 拼接所有数据
    X_combined = np.vstack(all_X_data)
    Y_combined = np.vstack(all_Y_data)

    print(f"\n合并后数据统计:")
    print(f"  总样本数: {X_combined.shape[0]}")
    print(f"  特征数: {X_combined.shape[1]}")
    print(f"  输入范围: [{X_combined.min():.3f}, {X_combined.max():.3f}]")
    print(f"  输出范围: [{Y_combined.min():.3f}, {Y_combined.max():.3f}]")

    # 添加噪声（如果需要）
    noise_info = None
    if noise_config is not None:
        print(f"\n添加噪声...")
        Y_combined, noise_info = add_noise_to_combined_data(Y_combined, **noise_config, seed=seed)

    # 统一归一化
    print(f"\n开始统一归一化...")
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_Y = MinMaxScaler(feature_range=(-1, 1))

    X_combined_scaled = scaler_X.fit_transform(X_normal)  # 此处有修改
    Y_combined_scaled = scaler_Y.fit_transform(Y_normal)

    print(f"归一化完成:")
    print(f"  X归一化后范围: [{X_combined_scaled.min():.3f}, {X_combined_scaled.max():.3f}]")
    print(f"  Y归一化后范围: [{Y_combined_scaled.min():.3f}, {Y_combined_scaled.max():.3f}]")

    # 划分训练集大小（仅来自正常数据）
    n_normal = len(X_normal)
    n_train = int(n_normal * training_rate)

    # 训练子集索引（保持最小改动：取正常数据前 n_train 个样本）
    # 如需随机子集，可参考下方“可选：随机抽样”说明
    train_indices = np.arange(n_train)

    # ===== 仅用“正常训练子集（原始值）”拟合 MinMaxScaler =====
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_Y = MinMaxScaler(feature_range=(-1, 1))
    scaler_X.fit(X_normal[train_indices])
    scaler_Y.fit(Y_normal[train_indices])

    # 合并所有数据（保持你的现有写法）
    X_combined = np.vstack(all_X_data)
    Y_combined = np.vstack(all_Y_data)

    # 仅做 transform（不要再 fit_transform）
    X_combined_scaled = scaler_X.transform(X_combined)
    Y_combined_scaled = scaler_Y.transform(Y_combined)

    # 训练/测试集构造（与原逻辑一致）
    x_train = X_combined_scaled[train_indices]
    y_train = Y_combined_scaled[train_indices]

    x_test = X_combined_scaled
    y_test = Y_combined_scaled
    print(f"\n最终数据集:")
    print(f"  训练集: {x_train.shape[0]} 样本 (全部来自正常数据)")
    print(f"  测试集: {x_test.shape[0]} 样本 (包含所有数据)")

    # 分析测试集构成
    print(f"\n测试集详细构成:")
    cumulative_count = 0

    # 正常数据
    normal_count = len(X_normal)
    print(
        f"  正常数据: {normal_count} 样本 ({normal_count / len(x_test) * 100:.1f}%) [索引 0-{normal_count - 1}]")
    cumulative_count += normal_count

    # 各个故障数据
    for X_fault, Y_fault, label in fault_data_list:
        fault_count = len(X_fault)
        start_idx = cumulative_count
        end_idx = cumulative_count + fault_count - 1
        print(
            f"  {label}: {fault_count} 样本 ({fault_count / len(x_test) * 100:.1f}%) [索引 {start_idx}-{end_idx}]")
        cumulative_count += fault_count

    # ===== 绘图可视化（仅第一个子图）=====
    print(f"\n生成可视化图表...")

    # 创建单个图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 准备数据索引
    total_samples = len(Y_combined)
    sample_indices = np.arange(total_samples)

    # 创建训练集掩码（在整个数据集中标记训练样本）
    train_mask = np.zeros(total_samples, dtype=bool)
    train_mask[train_indices] = True

    # 绘制训练集vs测试集（归一化前）
    ax.scatter(sample_indices[~train_mask], Y_combined[~train_mask],
               c='orange', s=5, alpha=0.7, label='测试集（包含所有数据）')
    ax.scatter(sample_indices[train_mask], Y_combined[train_mask],
               c='blue', s=8, alpha=0.9, label='训练集（正常数据子集）')

    # 添加数据集分界线
    boundary_lines = [normal_count]
    current_pos = normal_count
    for X_fault, Y_fault, label in fault_data_list:
        current_pos += len(X_fault)
        boundary_lines.append(current_pos)

    for i, boundary in enumerate(boundary_lines[:-1]):
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
        if i == 0:
            ax.text(boundary / 2, ax.get_ylim()[1] * 0.9, '正常数据',
                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 标注故障数据区域
    for i, (X_fault, Y_fault, label) in enumerate(fault_data_list):
        start_pos = boundary_lines[i]
        end_pos = boundary_lines[i + 1]
        mid_pos = (start_pos + end_pos) / 2
        ax.text(mid_pos, ax.get_ylim()[1] * 0.4, label,
                ha='center', fontsize=6, rotation=0,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

    ax.set_xlabel('数据点索引')
    ax.set_ylabel('输出变量（原始值）')
    ax.set_title('训练集vs测试集分布（归一化前）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 转换为张量
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # 组织数据信息
    data_info = {
        'data_labels': data_labels,
        'train_indices': train_indices,
        'normal_samples': n_normal,
        'fault_samples': len(X_combined) - n_normal,
        'X_combined': X_combined,
        'Y_combined': Y_combined,
        'Y_combined_scaled': Y_combined_scaled,
        'noise_info': noise_info,
        'fault_data_list': fault_data_list,
        'boundary_lines': boundary_lines
    }

    print(f"数据处理和可视化完成!")

    return (x_train, y_train, x_test, y_test, scaler_X, scaler_Y, data_info)


class DNN(torch.nn.Module):
    def __init__(self, p, logvar, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1
        self.p = p
        self.logvar = logvar
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

            layer_list.append(('dropout_%d' % i, torch.nn.Dropout(p=self.p)))

        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

        # 预测均值和方差的分支
        self.predict = torch.nn.Linear(layers[-2], layers[-1])
        # 增强方差预测网络 - 添加更多层来学习复杂的方差模式
        self.var_layers = torch.nn.Sequential(
            torch.nn.Linear(layers[-2], layers[-2] // 2),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.p),
            torch.nn.Linear(layers[-2] // 2, layers[-2] // 4),
            torch.nn.Tanh(),
            torch.nn.Linear(layers[-2] // 4, layers[-1])
        )

    def forward(self, x):
        features = self.layers(x)

        # 预测均值
        out = self.predict(features)

        # 预测方差（取决于输入特征）
        if self.logvar:
            # 使用更复杂的网络来预测方差
            logvar = self.var_layers(features)
            # 使用softplus确保方差为正，并添加小的下界
            logvar = F.softplus(logvar) + 1e-6
            # 转换为log方差
            logvar = torch.log(logvar)
        else:
            logvar = torch.zeros(out.size()).to(out.device)

        return out, logvar


class PhysicsInformedNN():
    def __init__(self, X, u, layers, x_scal, u_scal, p, logvar):
        # ↓ 负责载入数据，初始化Physi待辨识参数，初始化DNN

        # data
        self.x = X[:, 0:].clone().detach().requires_grad_(True).float().to(device)

        self.u = u.clone().detach().float().to(device)
        self.u_scal = u_scal
        self.x_scal = x_scal
        self.X = X
        # settings 初始值
        self.lambda_1 = torch.tensor([0.167897923477715], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([2.36682075851268e-06], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([2.43414469188443], requires_grad=True).to(device)
        self.lambda_4 = torch.tensor([1.0], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_3)
        self.lambda_4 = torch.nn.Parameter(self.lambda_4)

        # deep neural networks
        self.dnn = DNN(p, logvar, layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)
        self.dnn.register_parameter('lambda_3', self.lambda_3)
        self.dnn.register_parameter('lambda_3', self.lambda_4)

        # # 添加温度模型参数(复杂模型)
        # self.lambda_T1= torch.tensor([1.1175], device=device)  # 冷却液换热系数
        # self.lambda_T2= torch.tensor([9392.846], device=device)  # 电堆热质量
        # self.lambda_T3 = torch.tensor([0.0], device=device)  # 电堆比热容
        # self.lambda_T4= torch.tensor([498.13], device=device)  # 产热效率(sigmoid->0.5)

        # 添加温度模型参数(模型)
        self.lambda_T1 = torch.tensor([10.0], device=device)  # 冷却液换热系数
        self.lambda_T2 = torch.tensor([10.0], device=device)  # 电堆热质量
        self.lambda_T3 = torch.tensor([10.0], device=device)  # 电堆比热容
        self.lambda_T4 = torch.tensor([10.0], device=device)  # 产热效率
        self.lambda_T5 = torch.tensor([10.0], device=device)  # 产热效率

        self.lambda_T1 = torch.nn.Parameter(self.lambda_T1)
        self.lambda_T2 = torch.nn.Parameter(self.lambda_T2)
        self.lambda_T3 = torch.nn.Parameter(self.lambda_T3)
        self.lambda_T4 = torch.nn.Parameter(self.lambda_T4)
        self.lambda_T5 = torch.nn.Parameter(self.lambda_T5)

        # 在DNN中注册温度参数
        self.dnn.register_parameter('lambda_T1', self.lambda_T1)
        self.dnn.register_parameter('lambda_T2', self.lambda_T2)
        self.dnn.register_parameter('lambda_T3', self.lambda_T3)
        self.dnn.register_parameter('lambda_T4', self.lambda_T4)
        self.dnn.register_parameter('lambda_T5', self.lambda_T5)

        # 添加氢气化学计量比模型参数
        self.lambda_H1 = torch.tensor([5.0000], device=device)  # 压力影响系数
        self.lambda_H2 = torch.tensor([-1.559], device=device)  # 温度影响系数
        self.lambda_H3 = torch.tensor([197.715], device=device)  # 流量修正系数
        self.lambda_H4 = torch.tensor([1.20], device=device)  # 基准化学计量比

        self.lambda_H1 = torch.nn.Parameter(self.lambda_H1)
        self.lambda_H2 = torch.nn.Parameter(self.lambda_H2)
        self.lambda_H3 = torch.nn.Parameter(self.lambda_H3)
        self.lambda_H4 = torch.nn.Parameter(self.lambda_H4)

        # 在DNN中注册氢气参数
        self.dnn.register_parameter('lambda_H1', self.lambda_H1)
        self.dnn.register_parameter('lambda_H2', self.lambda_H2)
        self.dnn.register_parameter('lambda_H3', self.lambda_H3)
        self.dnn.register_parameter('lambda_H4', self.lambda_H4)

        # 添加氧气化学计量比模型参数
        self.lambda_O1 = torch.tensor([2.0], device=device)  # 基础过量比常量
        self.lambda_O2 = torch.tensor([0.5], device=device)  # 电流修正系数
        self.lambda_O3 = torch.tensor([200.0], device=device)  # 电流饱和阈值
        self.lambda_O4 = torch.tensor([1.0], device=device)  # 保留参数

        self.lambda_O1 = torch.nn.Parameter(self.lambda_O1)
        self.lambda_O2 = torch.nn.Parameter(self.lambda_O2)
        self.lambda_O3 = torch.nn.Parameter(self.lambda_O3)
        self.lambda_O4 = torch.nn.Parameter(self.lambda_O4)

        # 在DNN中注册氧气参数
        self.dnn.register_parameter('lambda_O1', self.lambda_O1)
        self.dnn.register_parameter('lambda_O2', self.lambda_O2)
        self.dnn.register_parameter('lambda_O3', self.lambda_O3)
        self.dnn.register_parameter('lambda_O4', self.lambda_O4)

    def net_u(self, x):
        # ↓ 输入数据、直接计算DNN的输出
        prediction, log_var = self.dnn(x)
        return prediction, log_var

    def net_f_O(self, X, x_scal):
        """
        氧气化学计量比约束计算函数
        参考氢气模型的处理方法，但需要考虑空气中氧气含量约21%
        """
        try:
            # 载入和参数并还原
            real_data = x_scal.inverse_transform(X.detach().cpu().numpy())
            real_data = torch.tensor(real_data, device=device, dtype=torch.float32)

            # 物理常数
            A_cell = torch.tensor([270], dtype=torch.float32, device=device)  # cm²
            F = torch.tensor([96485], dtype=torch.float32, device=device)  # C/mol
            N_cells = torch.tensor([5], dtype=torch.float32, device=device)
            V_molar_STP = torch.tensor([22.4], dtype=torch.float32, device=device)  # L/mol at STP
            O2_fraction = torch.tensor([0.21], dtype=torch.float32, device=device)  # 空气中氧气含量

            # 提取输入变量
            i_current = real_data[:, 0:1] / A_cell + 0.00001  # A/cm² (电流密度)
            air_flow_slpm = real_data[:, 7:8] + 1e-6  # 空气流量 [slpm]

            # ================= 电流计算 =================
            I_stack = i_current * A_cell  # A (串联电池堆总电流，不乘单体数)

            # ================= 理论氧气消耗量计算 =================
            # 阴极反应：O2 + 4H+ + 4e- → 2H2O
            # 每消耗1摩尔O2需要4摩尔电子
            # 每个单体的氧气消耗：I_stack / (4 * F) mol/s
            # 总氧气消耗：单体消耗 × 单体数量
            n_O2_rate_total = (I_stack * N_cells) / (4 * F)  # mol/s (总氧气消耗)
            Q_O2_theoretical_Ls = n_O2_rate_total * V_molar_STP  # L/s at STP
            Q_O2_theoretical_slpm = Q_O2_theoretical_Ls * 60  # slpm (标准氧气流量)

            # 防止除零
            epsilon = torch.tensor([1e-8], dtype=torch.float32, device=device)
            Q_O2_theoretical_slpm = torch.clamp(Q_O2_theoretical_slpm, min=epsilon)

            # ================= 目标氧气过量比计算（分段函数）=================
            lambda_O1 = self.lambda_O1  # 基础过量比常量
            lambda_O2 = self.lambda_O2  # 电流修正系数
            lambda_O3 = self.lambda_O3  # 电流饱和阈值

            # 电流饱和阈值
            I_threshold = torch.abs(lambda_O3)  # 阈值单位：A

            # 电流归一化（相对于典型工作电流）
            I_norm = I_stack / torch.tensor([100.0], device=device)  # 以100A为基准归一化

            # 分段函数：使用torch.where实现if-else逻辑
            I_threshold_norm = I_threshold / torch.tensor([100.0], device=device)

            target_excess_ratio = torch.where(
                I_stack <= I_threshold,
                lambda_O1 + lambda_O2 * I_norm,  # 线性区域
                lambda_O1 + lambda_O2 * I_threshold_norm  # 饱和区域（常数）
            )

            # 确保目标过量比在合理范围内
            target_excess_ratio = torch.clamp(target_excess_ratio, min=1.05, max=15.0)

            # ================= 实际氧气流量处理 =================
            # 从空气流量计算实际氧气流量
            # 空气中氧气含量约21%
            o2_flow_actual = air_flow_slpm * O2_fraction  # slpm (实际氧气流量)

            # ================= 计算实际过量比 =================
            # 实际过量比 = 实际氧气流量 / 理论最小氧气流量
            actual_excess_ratio = o2_flow_actual / Q_O2_theoretical_slpm

            # ================= 氧气化学计量比约束 =================
            # 约束：实际过量比应该接近目标过量比
            f_O2 = actual_excess_ratio - target_excess_ratio

            # 物理约束：过量比不能小于1
            ratio_penalty = torch.clamp(1.0 - actual_excess_ratio, min=0.0)
            f_O2 = f_O2 + ratio_penalty * 10.0

            return f_O2, actual_excess_ratio, target_excess_ratio, Q_O2_theoretical_slpm, o2_flow_actual

        except Exception as e:
            print(f"net_f_O计算错误: {e}")
            batch_size = X.shape[0]
            # 返回零张量避免程序崩溃
            zero_tensor = torch.zeros(batch_size, 1, device=device)
            return zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor

    def net_f_H(self, X, x_scal):
        """
        氢气化学计量比约束计算函数
        通过参数优化学习目标过量比和电流、温度、压力的关系
        实际流量与压力成正比，通过参数矫正
        氢气流量单位：slpm (标准升每分钟)
        """
        # 载入和参数并还原 - 参考net_f_V的写法
        real_data = x_scal.inverse_transform(X.detach().cpu().numpy())
        real_data = torch.tensor(real_data, device=device, dtype=torch.float32)

        # 物理常数 - 参考net_f_V中的定义
        A_cell = torch.tensor([270], dtype=torch.float32, device=device)  # 平方厘米，参考net_f_V
        F = torch.tensor([96485], dtype=torch.float32, device=device)  # 法拉第常数，参考net_f_V
        R = torch.tensor([8.314], dtype=torch.float32, device=device)  # 气体常数，参考net_f_V
        N_cells = torch.tensor([5], dtype=torch.float32, device=device)  # 电池单体数量，参考net_f_V

        # 提取输入变量 - 根据您的数据集结构调整索引
        i_current = real_data[:, 0:1] / A_cell + 0.00001  # 电流密度 [A/cm²]，参考net_f_V处理方式
        m_coolant = real_data[:, 1:2]  # 冷却液流量 [kg/s]
        T_in = real_data[:, 2:3]  # 入口温度 [°C]
        P_H2_gauge = real_data[:, 3:4]  # 氢气压力 [Pa]，参考net_f_V
        P_air_gauge = real_data[:, 4:5]  # 空气压力 [Pa]，参考net_f_V
        T_celsius = real_data[:, 5:6]  # 出口温度 [°C]
        h2_flow_raw = real_data[:, 6:7] + 1e-6  # 氢气流量 [slpm]
        air_flow = real_data[:, 7:8]  # 空气流量 [slpm]

        # 温度和压力转换 - 参考net_f_V的处理方式
        T_kelvin = T_celsius + torch.tensor([273.15], dtype=torch.float32, device=device)
        P_H2 = P_H2_gauge / 101325 + 1  # 转换为相对压力 (注意单位转换)
        P_air = P_air_gauge / 101325 + 1  # 转换为相对压力

        # 计算总电流 [A]
        I_total = i_current * A_cell

        # ================= 理论氢气消耗量计算 =================
        # 根据法拉第定律：每消耗1摩尔H2产生2摩尔电子
        # I = n_H2 * 2 * F / t  =>  n_H2 = I * t / (2 * F)
        # 氢气消耗速率 [mol/s] = I / (2 * F)
        n_H2_rate = I_total / (2 * F) * N_cells  # [mol/s]

        # 标准状态下氢气体积流率 [L/s] = n_H2_rate * 22.4 [L/mol]
        # 但需要考虑温度和压力修正到实际条件
        V_H2_std = torch.tensor([22.4], dtype=torch.float32, device=device)  # 标准摩尔体积 [L/mol]

        # 理论氢气体积流率 [L/s]，在标准条件下
        Q_H2_theoretical_ls = n_H2_rate * V_H2_std  # [L/s] at STP

        # 转换为 [slpm] (标准升每分钟)
        Q_H2_theoretical = Q_H2_theoretical_ls * 60  # [slpm]

        # 添加小的epsilon防止除零
        epsilon = torch.tensor([1e-8], dtype=torch.float32, device=device)
        Q_H2_theoretical = torch.clamp(Q_H2_theoretical, min=epsilon)

        # ================= 目标氢气过量比计算（分段函数）=================
        # 参数说明：
        # lambda_H1: 基础过量比常量 (范围: 1.1 - 5.0)
        # lambda_H2: 电流修正系数 (范围: -1.0 - 1.0)
        # lambda_H3: 电流饱和阈值 [A] (范围: 50 - 200)
        # lambda_H4: 暂时保留，可用于其他修正
        lambda_H1 = self.lambda_H1  # 电流对温升的影响系数
        lambda_H2 = self.lambda_H2  # 流量对温降的影响系数
        lambda_H3 = self.lambda_H3  # 热效率系数
        lambda_H4 = self.lambda_H4  # 基准温度偏移
        # 电流饱和阈值
        I_threshold = lambda_H3  # 阈值范围：50-250A

        # 电流归一化（相对于典型工作电流）
        I_norm = I_total / torch.tensor([100.0], device=device)  # 以100A为基准归一化

        # 分段函数：使用torch.where实现if-else逻辑
        # 当 I_total <= I_threshold 时：target_ratio = lambda_H1 + lambda_H2 * I_norm
        # 当 I_total > I_threshold 时：target_ratio = lambda_H1 + lambda_H2 * (I_threshold/100)
        I_threshold_norm = I_threshold / torch.tensor([100.0], device=device)

        target_excess_ratio = torch.where(
            I_total <= I_threshold,
            lambda_H1 + lambda_H2 * I_norm,  # 线性区域
            lambda_H1 + lambda_H2 * I_threshold_norm  # 饱和区域（常数）
        )

        # 确保目标过量比在合理范围内
        # target_excess_ratio = torch.clamp(target_excess_ratio, min=1.05, max=10.0)

        # ================= 实际氢气流量处理（与压力成正比，参数矫正） =================
        # 实际流量已经是slpm，直接使用
        # 但考虑压力对流量测量的影响，通过参数矫正

        # 实际氢气流量：考虑压力修正
        h2_flow_actual = h2_flow_raw

        # ================= 计算实际过量比 =================
        # 实际过量比 = 实际流量 / 理论最小流量
        actual_excess_ratio = h2_flow_actual / Q_H2_theoretical

        # ================= 氢气化学计量比约束 =================
        # 约束：实际过量比应该接近目标过量比
        f_H2 = actual_excess_ratio - target_excess_ratio

        # ================= 返回结果 =================
        return f_H2, actual_excess_ratio, target_excess_ratio, I_total, I_threshold

    def net_f_V(self, X, x_scal):
        x_in = X[:, 0:].clone().detach().requires_grad_(True).float().to(device)
        real_data = x_scal.inverse_transform(X.detach().cpu().numpy())
        real_data = torch.tensor(real_data, device=device)

        A_cell = torch.tensor([270], dtype=torch.float32, device=device)
        i = real_data[:, 0:1] / A_cell + 1e-5
        T_out = real_data[:, 5:6]

        u, log_var = self.net_u(x_in)
        u_cpu = u.detach().cpu().numpy()
        V_out = self.u_scal.inverse_transform(u_cpu)
        N_cells = torch.tensor([5], dtype=torch.float32, device=device)
        V_out = torch.tensor(V_out, device=device) / N_cells

        r = self.lambda_1
        io = self.lambda_2
        il = self.lambda_3

        R = torch.tensor([8.314], dtype=torch.float32, device=device)
        F = torch.tensor([96485], dtype=torch.float32, device=device)
        Tc = torch.tensor([55], dtype=torch.float32, device=device)
        P_H2 = real_data[:, 3:4] / 101 + 1
        P_air = real_data[:, 4:5] / 101 + 1
        Alpha = torch.tensor([0.5], dtype=torch.float32, device=device)
        Gf_liq = torch.tensor([-220170], dtype=torch.float32, device=device)
        Tk = T_out + torch.tensor([273.15], dtype=torch.float32, device=device)

        x = -2.1794 + 0.02953 * Tc - 9.1837e-5 * (Tc ** 2) + 1.4454e-7 * (Tc ** 3)
        P_H2O = (10 ** x)
        pp_H2 = 0.5 * ((P_H2) / (torch.exp(1.653 * i / (Tk ** 1.334))) - P_H2O)
        pp_O2 = (P_air / torch.exp(4.192 * i / (Tk ** 1.334))) - P_H2O
        b = R * Tk / (2. * Alpha * F)

        V_act = -b * torch.log(i / io)
        V_ohmic = -(i * r)
        V_conc = Alpha * b * torch.log(1 - (i / il))
        E_nerst = -Gf_liq / (2 * F) - ((R * Tk) * torch.log(P_H2O / (pp_H2 * (pp_O2 ** 0.5)))) / (2 * F)
        V_out_est = E_nerst + V_act + V_ohmic + V_conc

        f = V_out_est - V_out
        return f, V_act, V_ohmic, V_conc, E_nerst, V_out_est * 5, i, il, V_out * 5

    def net_f_T(self, X, x_scal):
        """
        简化的时序温度模型
        基于1到N-1的温度预测2到N的温度，并构建预测序列
        """
        batch_size = X.shape[0]

        if batch_size < 2:
            # 数据不足，返回零约束
            return (torch.zeros(batch_size, 1, device=device),
                    torch.zeros(batch_size, 1, device=device),
                    torch.zeros(batch_size, 1, device=device))

        # 获取真实物理量 - 参考 net_f_V 的写法
        real_data = x_scal.inverse_transform(X.detach().cpu().numpy())
        real_data = torch.tensor(real_data, device=device, dtype=torch.float32)

        # 提取变量
        A_cell = torch.tensor([270], dtype=torch.float32, device=device)  # 参考 net_f_V
        i_current = real_data[:, 0:1] / A_cell + 0.00001  # 电流密度 [A/cm²]，参考 net_f_V
        m_coolant = real_data[:, 1:2] + 1e-6  # 冷却液流量 [kg/s]
        T_in = real_data[:, 2:3]  # 入口温度 [°C]
        T_out = real_data[:, 5:6]  # 出口温度（电堆温度）[°C]

        # 物理常数 - 确保都在正确设备上
        F = torch.tensor([96485], dtype=torch.float32, device=device)  # 参考 net_f_V
        N_cells = torch.tensor([5], dtype=torch.float32, device=device)  # 参考 net_f_V
        cp_coolant = torch.tensor([4180.0], dtype=torch.float32, device=device)
        dt = torch.tensor([0.1], dtype=torch.float32, device=device)
        h_air = torch.tensor([20.0], dtype=torch.float32, device=device)
        A_surface = torch.tensor([0.2], dtype=torch.float32, device=device)
        T_ambient = torch.tensor([25.0], dtype=torch.float32, device=device)

        # 热力学参数（待辨识）
        alpha_coolant = self.lambda_T1  # 冷却液换热系数 [W/(m²·K)]
        m_stack = self.lambda_T2  # 电堆热质量 [kg]
        cp_stack = self.lambda_T3  # 电堆比热容 [J/(kg·K)]
        eta_heat = self.lambda_T4  # 产热效率 [0-1]

        # ================= 基于前N-1个点预测后N-1个点 =================

        # 提取前N-1个时刻的数据用于预测
        i_prev = i_current[:-1]  # [0 to N-2]
        m_coolant_prev = m_coolant[:-1]
        T_in_prev = T_in[:-1]
        T_out_prev = T_out[:-1]  # 当前时刻电堆温度

        # 计算总电流 - 修正单位
        I_total_prev = i_prev * A_cell  # [A] - 参考 net_f_V 中的 A_cell 处理

        # 1. 电化学产热功率 [W]
        # 理论电压（简化的温度依赖）
        T_kelvin_prev = T_out_prev + 273.15
        V_reversible_prev = 1.229 - 0.0009 * (T_kelvin_prev - 298.15)

        # 获取实际电压（调用电压预测）- 修正设备问题
        # 创建输入张量，确保在正确设备上
        X_prev = X[:-1, :].clone().detach().requires_grad_(True).float().to(device)

        # 调用电压预测 - 参考 net_f_V 的写法
        u_pred_prev, log_var_prev = self.net_u(X_prev)

        # 转换电压预测结果 - 参考 net_f_V 的处理方式
        u_cpu_prev = u_pred_prev.detach().cpu().numpy()
        V_actual_total_prev = self.u_scal.inverse_transform(u_cpu_prev)
        V_actual_total_prev = torch.tensor(V_actual_total_prev, device=device, dtype=torch.float32)
        V_actual_single_prev = V_actual_total_prev / N_cells  # 单电池电压 [V]

        # 电化学产热 = 总输入功率 - 实际输出功率
        P_input_prev = I_total_prev * V_reversible_prev  # 理论输入功率
        P_output_prev = I_total_prev * V_actual_single_prev  # 实际输出功率
        Q_electrochemical_prev = (P_input_prev - P_output_prev) * eta_heat

        # 2. 冷却液带走的热量 [W]
        Q_coolant_prev = m_coolant_prev * cp_coolant * (T_out_prev - T_in_prev) * alpha_coolant

        # 3. 空气辐射散热 [W]
        Q_radiation_prev = h_air * A_surface * (T_out_prev - T_ambient) * cp_stack

        # 4. 电堆温度变化率 [°C/s]
        # 能量平衡：dT/dt = (产热 - 散热) / (热质量 × 比热容)
        Q_net_prev = Q_electrochemical_prev - Q_coolant_prev - Q_radiation_prev
        dT_dt_prev = Q_net_prev / (m_stack)

        # 5. 预测下一时刻温度（欧拉积分）
        T_out_predicted_next = T_out_prev + dT_dt_prev * dt

        # ================= 构建完整的预测序列 =================

        # 将第一个真实温度拼接到预测序列前面
        T_out_predicted_full = torch.cat([T_out[0:1], T_out_predicted_next], dim=0)

        # 真实温度序列就是T_out
        T_out_real_full = T_out

        # ================= 计算温度约束 =================

        # 温度预测误差作为物理约束
        f_T_residual =  T_out_real_full - T_out_predicted_full

        return f_T_residual, T_out_predicted_full, T_out_real_full

    def net_f_T_simple(self, X, x_scal):
        """
        超简化的温度预测函数：基于线性关系
        """
        x_in = X[:, 0:].clone().detach().float().to(device)
        u, log_var = self.net_u(x_in)
        u_cpu = u.detach().cpu().numpy()
        V_out = self.u_scal.inverse_transform(u_cpu)
        V_out = torch.tensor(V_out, device=device, dtype=torch.float32)
        # 载入和参数并还原
        real_data = x_scal.inverse_transform(X.detach().cpu().numpy())
        real_data = torch.tensor(real_data, device=device, dtype=torch.float32)

        # 提取输入变量
        A_cell = torch.tensor(270.0, dtype=torch.float32, device=device)
        i = real_data[:, 0:1] / A_cell + 1e-6  # 电流密度
        m_coolant = real_data[:, 1:2] + 1e-6  # 冷却液流量，防止除零
        T_in = real_data[:, 2:3]  # 入口温度
        T_out_real = real_data[:, 5:6]  # 真实出口温度

        # 温度模型参数
        lambda_T1 = self.lambda_T1  # 电流对温升的影响系数
        lambda_T2 = self.lambda_T2  # 流量对温降的影响系数
        lambda_T3 = self.lambda_T3  # 热效率系数
        lambda_T4 = self.lambda_T4  # 基准温度偏移
        lambda_T5 = self.lambda_T5  # 基准温度偏移

        # 超简单的线性温度模型
        # T_out = T_in + (电流引起的温升) - (流量引起的温降) + 偏移
        I_total = i * A_cell  # 总电流

        # 预测温度
        # T_out_predicted = T_in + temp_rise - temp_drop + lambda_T4
        # T_out_predicted = (lambda_T1 * I_total* (lambda_T2-V_out)) / (lambda_T3 * m_coolant) + lambda_T4 * T_in + lambda_T5

        # T_out_predicted = lambda_T1 * I_total + lambda_T2 * V_out + lambda_T3 * m_coolant + lambda_T4 * T_in + lambda_T5
        T_out_predicted = lambda_T1 * I_total + lambda_T3 * m_coolant + 0.5 * T_in + lambda_T5
        # 温度约束在合理范围内
        # T_out_predicted = torch.clamp(T_out_predicted,
        #                               min=T_in - 5.0,  # 最多比入口温度低5度
        #                               max=T_in + 30.0)  # 最多比入口温度高30度

        # 计算温度预测残差
        f_T = T_out_real - T_out_predicted

        return f_T, T_out_predicted, T_out_real

    def aleatoric_loss(self, gt, pred_y, logvar):
        """异方差损失函数 - 能够学习输入相关的噪声"""
        # 标准的异方差损失 - 理论基础更扎实
        precision = torch.exp(-logvar)  # 精度 = 1/方差

        # 负对数似然损失（基于高斯分布假设）
        loss = torch.mean(0.5 * precision * (gt - pred_y) ** 2 + 0.5 * logvar)

        # 添加方差的正则化项，防止方差过小或过大
        var_reg = torch.mean(torch.abs(logvar))  # L1正则化

        return loss + 0.01 * var_reg  # 权重可调

    def train_dnn(self, nIter):
        """ 仅训练DNN参数 """
        # 解冻DNN参数，冻结lambda参数
        for param in self.dnn.parameters():
            param.requires_grad = True
        self.lambda_1.requires_grad = False
        self.lambda_2.requires_grad = False
        self.lambda_3.requires_grad = False
        self.lambda_4.requires_grad = False

        optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=0.01)
        scheduler_Adam = StepLR(optimizer_Adam, step_size=1000, gamma=0.8, verbose=False)

        print('================== DNN训练 ==================')
        print('  Epoch |    Loss    |    MSE     |    LR    ')
        print('--------|------------|------------|----------')

        self.dnn.train()

        for epoch in range(nIter):
            u_pred, log_var = self.net_u(self.x)
            loss = self.aleatoric_loss(self.u, u_pred, log_var)

            optimizer_Adam.zero_grad()
            loss.backward()
            optimizer_Adam.step()
            scheduler_Adam.step()

            if epoch % 1000 == 0:
                lr_current = optimizer_Adam.param_groups[0]['lr']
                mse = torch.mean((self.u - u_pred) ** 2).item()

                print(f' {epoch:5d}  | {loss.item():10.3e} | {mse:10.3e} | {lr_current:8.1e}')

        print(f'DNN训练完成，最终Loss: {loss.item():.3e}')
        print()

    def train_lambda(self, nIter, dnn_para=False):
        self.dnn.eval()
        for param in self.dnn.parameters():
            param.requires_grad = dnn_para
        # 冻结温度模型参数
        self.lambda_T1.requires_grad = False
        self.lambda_T2.requires_grad = False
        self.lambda_T3.requires_grad = False
        self.lambda_T4.requires_grad = False
        self.lambda_T5.requires_grad = False

        self.lambda_O1.requires_grad = False
        self.lambda_O2.requires_grad = False
        self.lambda_O3.requires_grad = False
        self.lambda_O4.requires_grad = False
        # 解冻氢气参数
        self.lambda_H1.requires_grad = False
        self.lambda_H2.requires_grad = False
        self.lambda_H3.requires_grad = False
        self.lambda_H4.requires_grad = False

        self.lambda_1.requires_grad = True
        self.lambda_2.requires_grad = True
        self.lambda_3.requires_grad = True
        self.lambda_4.requires_grad = True

        param_bounds = {
            'lambda_1': [0.167 * 0.5, 0.167 * 5],
            'lambda_2': [2.36e-6 * 0.1, 2.36e-6 * 2.1],
            'lambda_3': [2.0, 2.0 * 5.2],
            'lambda_4': [0.1, 10.0]
        }

        optimizer_lambda = torch.optim.Adam(
            [self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4], lr=1e-3
        )
        scheduler_lambda = StepLR(optimizer_lambda, step_size=1000, gamma=0.8)

        print('================ 电压参数训练 ================')
        print('  Epoch |  总Loss   |  物理Loss  |   λ1    |    λ2     |   λ3   |    LR    ')
        print('--------|-----------|------------|---------|-----------|--------|----------')

        for epoch in range(nIter):
            u_pred, *_ = self.net_u(self.x)
            # 先得到物理域的堆栈电压估计（已乘5）
            f_pred, V_act, V_ohmic, V_conc, E_nerst, V_out_est, i, il, V_out = self.net_f_V(self.X, self.x_scal)

            # 将物理域的 V_out_est 映射回标准化域（与 self.u 同域）
            # MinMaxScaler: y_norm = y_physical * scale + min

            # 读取 MinMaxScaler 的仿射参数并搬到正确的设备/类型
            lo_y = float(self.u_scal.feature_range[0])
            hi_y = float(self.u_scal.feature_range[1])
            data_min_y = torch.tensor(self.u_scal.data_min_, dtype=torch.float32, device=self.u.device)  # [1]
            data_max_y = torch.tensor(self.u_scal.data_max_, dtype=torch.float32, device=self.u.device)  # [1]
            scale_y = (hi_y - lo_y) / (data_max_y - data_min_y + 1e-12)  # 标量张量
            min_y = lo_y - data_min_y * scale_y  # 标量张量

            # 注意：V_out_est 的形状是 [N,1]，已在物理域（单位：V，堆栈）
            V_out_est_norm = V_out_est * scale_y + min_y

            if dnn_para:
                # 第二阶段：物理残差 f^2（这个本来就在标准化域内无关）
                physics_loss = torch.mean(f_pred ** 2)
            else:
                # 第一阶段（PM）：用标准化域的一致损失
                physics_loss = torch.mean((self.u - V_out_est_norm) ** 2)
            data_loss = torch.mean((self.u - u_pred) ** 2)
            total_loss = physics_loss + data_loss

            optimizer_lambda.zero_grad()
            total_loss.backward()
            optimizer_lambda.step()

            self.lambda_1.data = torch.clamp(self.lambda_1.data, param_bounds['lambda_1'][0],
                                             param_bounds['lambda_1'][1])
            self.lambda_2.data = torch.clamp(self.lambda_2.data, param_bounds['lambda_2'][0],
                                             param_bounds['lambda_2'][1])
            self.lambda_3.data = torch.clamp(self.lambda_3.data, param_bounds['lambda_3'][0],
                                             param_bounds['lambda_3'][1])
            self.lambda_4.data = torch.clamp(self.lambda_4.data, param_bounds['lambda_4'][0],
                                             param_bounds['lambda_4'][1])

            if epoch % 1000 == 0:
                lr_current = optimizer_lambda.param_groups[0]['lr']
                print(f' {epoch:5d}  | {total_loss.item():9.3e} | {physics_loss.item():10.3e} | '
                      f'{self.lambda_1.item():7.4f} | {self.lambda_2.item():9.2e} | '
                      f'{self.lambda_3.item():6.3f} | {lr_current:8.1e}')

            scheduler_lambda.step()

        print(f'电压参数训练完成，最终Loss: {total_loss.item():.3e}')
        print()

    def train_thermal(self, nIter):
        """ 专门训练温度模型的热参数 """
        # 冻结DNN参数和电压lambda参数
        self.dnn.eval()
        for param in self.dnn.parameters():
            param.requires_grad = False

        # 冻结电压模型参数
        self.lambda_1.requires_grad = False
        self.lambda_2.requires_grad = False
        self.lambda_3.requires_grad = False
        self.lambda_4.requires_grad = False

        self.lambda_O1.requires_grad = False
        self.lambda_O2.requires_grad = False
        self.lambda_O3.requires_grad = False
        self.lambda_O4.requires_grad = False
        # 解冻氢气参数
        self.lambda_H1.requires_grad = False
        self.lambda_H2.requires_grad = False
        self.lambda_H3.requires_grad = False
        self.lambda_H4.requires_grad = False
        # 解冻温度参数
        self.lambda_T1.requires_grad = True
        self.lambda_T2.requires_grad = True
        self.lambda_T3.requires_grad = True
        self.lambda_T4.requires_grad = True
        self.lambda_T5.requires_grad = True
        # 定义温度参数边界
        thermal_param_bounds = {
            'lambda_T1': [-10000, 10000],  # 热传递系数
            'lambda_T2': [-10000, 10000],  # 热容系数
            'lambda_T3': [-10000, 10000],  # 热效率系数
            'lambda_T4': [-10000, 10000],  # 环境温度补偿
            'lambda_T5': [-10000, 10000]  # 环境温度补偿
        }

        # 单独优化温度参数
        optimizer_thermal = torch.optim.Adam([
            self.lambda_T1, self.lambda_T2, self.lambda_T3, self.lambda_T4, self.lambda_T5
        ], lr=1)

        scheduler_thermal = StepLR(optimizer_thermal, step_size=1000, gamma=0.8)
        print('----------------热参数训练阶段----------------')
        print(' Epoch |   Loss    |  MAE(°C) |   T1    |   T2   |   T3   |   T4   |    T5    |    LR   |')
        print('-------|-----------|----------|---------|--------|--------|--------|----------')

        for epoch in range(nIter):
            # 只计算温度约束，不包含电压约束
            f_T_pred, T_out_predicted, T_out_real = self.net_f_T_simple(self.X, self.x_scal)

            # 温度损失函数
            thermal_loss = torch.mean(f_T_pred ** 2)

            # 可选：添加温度预测的平滑性约束
            # if len(T_out_predicted) > 1:
            #     # 温度变化率约束（防止温度突变）
            #     dT_dt = T_out_predicted[1:] - T_out_predicted[:-1]
            #     smooth_loss = torch.mean(dT_dt ** 2)
            #     thermal_loss += 0.01 * smooth_loss  # 平滑性权重

            # Backward and optimize
            optimizer_thermal.zero_grad()
            thermal_loss.backward()
            optimizer_thermal.step()

            # **温度参数边界约束**
            self.lambda_T1.data = torch.clamp(self.lambda_T1.data,
                                              min=thermal_param_bounds['lambda_T1'][0],
                                              max=thermal_param_bounds['lambda_T1'][1])
            self.lambda_T2.data = torch.clamp(self.lambda_T2.data,
                                              min=thermal_param_bounds['lambda_T2'][0],
                                              max=thermal_param_bounds['lambda_T2'][1])
            self.lambda_T3.data = torch.clamp(self.lambda_T3.data,
                                              min=thermal_param_bounds['lambda_T3'][0],
                                              max=thermal_param_bounds['lambda_T3'][1])
            self.lambda_T4.data = torch.clamp(self.lambda_T4.data,
                                              min=thermal_param_bounds['lambda_T4'][0],
                                              max=thermal_param_bounds['lambda_T4'][1])
            self.lambda_T5.data = torch.clamp(self.lambda_T5.data,
                                              min=thermal_param_bounds['lambda_T5'][0],
                                              max=thermal_param_bounds['lambda_T5'][1])
            if epoch % 1000 == 0:
                lr_current = optimizer_thermal.param_groups[0]['lr']
                temp_mae = torch.mean(torch.abs(f_T_pred)).item()

                print(f' {epoch:3d}   | {thermal_loss.item():9.3e} | {temp_mae:8.2f} | '
                      f'{self.lambda_T1.item():7.4f} | {self.lambda_T2.item():6.3f} | '
                      f'{self.lambda_T3.item():6.3f} | {self.lambda_T4.item():6.2f} | {self.lambda_T5.item():6.2f} |'
                      f'{lr_current:8.1e}')

            scheduler_thermal.step()

    def train_oxygen(self, nIter):
        """ 专门训练氧气模型的参数 """
        # 冻结其他参数
        self.dnn.eval()
        for param in self.dnn.parameters():
            param.requires_grad = False

        self.lambda_1.requires_grad = False
        self.lambda_2.requires_grad = False
        self.lambda_3.requires_grad = False
        self.lambda_4.requires_grad = False

        self.lambda_T1.requires_grad = False
        self.lambda_T2.requires_grad = False
        self.lambda_T3.requires_grad = False
        self.lambda_T4.requires_grad = False
        self.lambda_T5.requires_grad = False

        self.lambda_H1.requires_grad = False
        self.lambda_H2.requires_grad = False
        self.lambda_H3.requires_grad = False
        self.lambda_H4.requires_grad = False

        # 解冻氧气参数
        self.lambda_O1.requires_grad = True
        self.lambda_O2.requires_grad = True
        self.lambda_O3.requires_grad = True
        self.lambda_O4.requires_grad = True

        # 定义氧气参数边界
        oxygen_param_bounds = {
            'lambda_O1': [1.5, 8.0],  # 基础过量比（氧气通常比氢气过量比大）
            'lambda_O2': [-20.0, 20.0],  # 电流修正系数
            'lambda_O3': [50, 1000],  # 电流饱和阈值
            'lambda_O4': [0.0, 20.0]  # 保留参数
        }

        # 单独优化氧气参数
        optimizer_oxygen = torch.optim.Adam([
            self.lambda_O1, self.lambda_O2, self.lambda_O3, self.lambda_O4
        ], lr=1e-2)

        scheduler_oxygen = StepLR(optimizer_oxygen, step_size=1000, gamma=0.9)

        print('================氧气参数训练阶段================')
        print(' Epoch |   Loss    | 实际过量比 | 目标过量比 |   O1    |   O2   |   O3   |   O4   |    LR    ')
        print('-------|-----------|------------|------------|---------|--------|--------|--------|----------')

        # 初始化损失变量
        oxygen_loss = torch.tensor(float('inf'), device=device)

        for epoch in range(nIter):
            try:
                # 计算氧气约束
                result = self.net_f_O(self.X, self.x_scal)

                if len(result) != 5:  # 期望返回5个值
                    print(f"警告：net_f_O返回了{len(result)}个值，期望5个值")
                    if len(result) >= 3:
                        f_O_pred, actual_ratio, target_ratio = result[:3]
                        Q_theoretical = torch.zeros_like(f_O_pred)  # 占位符
                        o2_actual = torch.zeros_like(f_O_pred)  # 占位符
                    else:
                        print(f"返回值数量不足，跳过epoch {epoch}")
                        continue
                else:
                    f_O_pred, actual_ratio, target_ratio, Q_theoretical, o2_actual = result

                # 氧气损失函数
                oxygen_loss = torch.mean(f_O_pred ** 2)

                # 反向传播
                optimizer_oxygen.zero_grad()
                oxygen_loss.backward()
                optimizer_oxygen.step()

                # 参数约束
                self.lambda_O1.data = torch.clamp(self.lambda_O1.data,
                                                  min=oxygen_param_bounds['lambda_O1'][0],
                                                  max=oxygen_param_bounds['lambda_O1'][1])
                self.lambda_O2.data = torch.clamp(self.lambda_O2.data,
                                                  min=oxygen_param_bounds['lambda_O2'][0],
                                                  max=oxygen_param_bounds['lambda_O2'][1])
                self.lambda_O3.data = torch.clamp(self.lambda_O3.data,
                                                  min=oxygen_param_bounds['lambda_O3'][0],
                                                  max=oxygen_param_bounds['lambda_O3'][1])
                self.lambda_O4.data = torch.clamp(self.lambda_O4.data,
                                                  min=oxygen_param_bounds['lambda_O4'][0],
                                                  max=oxygen_param_bounds['lambda_O4'][1])

                if epoch % 1000 == 0:
                    lr_current = optimizer_oxygen.param_groups[0]['lr']
                    actual_mean = torch.mean(actual_ratio).item()
                    target_mean = torch.mean(target_ratio).item()

                    # 计算实际电流饱和阈值
                    current_threshold = torch.abs(self.lambda_O3).item()

                    print(f' {epoch:3d}   | {oxygen_loss.item():6.3e} | {actual_mean:6.3f} | {target_mean:6.3f} | '
                          f'{self.lambda_O1.item():7.2f} | {self.lambda_O2.item():6.3f} | '
                          f'{current_threshold:6.1f}A | {self.lambda_O4.item():5.2f} | '
                          f'{lr_current:8.1e}')

                    # 额外输出详细信息
                    if epoch % 5000 == 0 and epoch > 0:
                        # 计算电流统计
                        real_data = self.x_scal.inverse_transform(self.X.detach().cpu().numpy())
                        current_data = real_data[:, 0]  # 电流列
                        print(f"    详细信息 - 数据电流范围: [{current_data.min():.1f}, {current_data.max():.1f}]A")
                        print(f"    平均电流: {current_data.mean():.1f}A, 饱和阈值: {current_threshold:.1f}A")

                        # 分析有多少数据在线性区域
                        A_cell = 270
                        i_current = torch.tensor(current_data, device=device) / A_cell
                        I_total = i_current * A_cell  # 总电流（修正后不乘单体数）
                        linear_mask = I_total <= current_threshold
                        n_linear = torch.sum(linear_mask).item()
                        print(f"    线性区域样本: {n_linear}/{len(I_total)} ({n_linear / len(I_total) * 100:.1f}%)")
                        print(
                            f"    饱和区域样本: {len(I_total) - n_linear}/{len(I_total)} ({(len(I_total) - n_linear) / len(I_total) * 100:.1f}%)")

                scheduler_oxygen.step()

            except Exception as e:
                print(f"氧气约束计算失败 (epoch {epoch}): {e}")
                print(f"错误类型: {type(e).__name__}")

                # 打印更详细的错误信息
                import traceback
                print("详细错误堆栈:")
                traceback.print_exc()

                # 如果是计算错误，跳过这次迭代
                continue

        print(f'氧气参数训练完成')

        # 输出最终参数解释
        final_threshold = torch.abs(self.lambda_O3).item()
        final_saturated_ratio = self.lambda_O1.item() + self.lambda_O2.item() * (final_threshold / 100)

        print(f'最终氧气参数解释:')
        print(f'  基础过量比: {self.lambda_O1.item():.3f}')
        print(f'  电流修正系数: {self.lambda_O2.item():.3f}')
        print(f'  电流饱和阈值: {final_threshold:.1f}A')
        print(f'  饱和区域目标过量比: {final_saturated_ratio:.3f}')
        print(f'  分段函数:')
        print(
            f'    I ≤ {final_threshold:.1f}A: 过量比 = {self.lambda_O1.item():.3f} + {self.lambda_O2.item():.3f} × (I/100)')
        print(f'    I > {final_threshold:.1f}A: 过量比 = {final_saturated_ratio:.3f}')
        print()

    def train_hydrogen(self, nIter):
        """ 专门训练氢气模型的参数 """
        # 冻结DNN参数和其他物理参数
        self.dnn.eval()
        for param in self.dnn.parameters():
            param.requires_grad = False

        # 冻结电压模型参数
        self.lambda_1.requires_grad = False
        self.lambda_2.requires_grad = False
        self.lambda_3.requires_grad = False
        self.lambda_4.requires_grad = False

        # 冻结温度模型参数
        self.lambda_T1.requires_grad = False
        self.lambda_T2.requires_grad = False
        self.lambda_T3.requires_grad = False
        self.lambda_T4.requires_grad = False
        self.lambda_T5.requires_grad = False

        self.lambda_O1.requires_grad = False
        self.lambda_O2.requires_grad = False
        self.lambda_O3.requires_grad = False
        self.lambda_O4.requires_grad = False
        # 解冻氢气参数
        self.lambda_H1.requires_grad = True
        self.lambda_H2.requires_grad = True
        self.lambda_H3.requires_grad = True
        self.lambda_H4.requires_grad = True

        # 定义氢气参数边界（重新定义含义）
        hydrogen_param_bounds = {
            'lambda_H1': [0.5, 50.0],  # 基础过量比
            'lambda_H2': [-20, 20],  # 电流修正系数（归一化电流）
            'lambda_H3': [50, 1000],  # 电流饱和阈值
            'lambda_H4': [0.0, 20.0]  # 保留
        }

        # 单独优化氢气参数
        optimizer_hydrogen = torch.optim.Adam([
            self.lambda_H1, self.lambda_H2, self.lambda_H3, self.lambda_H4
        ], lr=1e-1)

        scheduler_hydrogen = StepLR(optimizer_hydrogen, step_size=1000, gamma=0.9)

        print('================氢气参数训练阶段================')
        print(' Epoch |   Loss    | 实际过量比 | 目标过量比 |   H1    |   H2   |   H3   |   H4   |    LR    ')
        print('-------|-----------|------------|------------|---------|--------|--------|--------|----------')

        for epoch in range(nIter):
            try:
                # 计算氢气约束（现在返回5个值）
                f_H_pred, actual_ratio, target_ratio, *_ = self.net_f_H(self.X, self.x_scal)

                # 氢气损失函数
                hydrogen_loss = torch.mean(f_H_pred ** 2)

                # Backward and optimize
                optimizer_hydrogen.zero_grad()
                hydrogen_loss.backward()
                optimizer_hydrogen.step()

                # **氢气参数边界约束**
                self.lambda_H1.data = torch.clamp(self.lambda_H1.data,
                                                  min=hydrogen_param_bounds['lambda_H1'][0],
                                                  max=hydrogen_param_bounds['lambda_H1'][1])
                self.lambda_H2.data = torch.clamp(self.lambda_H2.data,
                                                  min=hydrogen_param_bounds['lambda_H2'][0],
                                                  max=hydrogen_param_bounds['lambda_H2'][1])
                self.lambda_H3.data = torch.clamp(self.lambda_H3.data,
                                                  min=hydrogen_param_bounds['lambda_H3'][0],
                                                  max=hydrogen_param_bounds['lambda_H3'][1])
                self.lambda_H4.data = torch.clamp(self.lambda_H4.data,
                                                  min=hydrogen_param_bounds['lambda_H4'][0],
                                                  max=hydrogen_param_bounds['lambda_H4'][1])

                if epoch % 1000 == 0:
                    lr_current = optimizer_hydrogen.param_groups[0]['lr']
                    actual_mean = torch.mean(actual_ratio).item()
                    target_mean = torch.mean(target_ratio).item()

                    print(f' {epoch:3d}   | {hydrogen_loss.item():9.3e} | {actual_mean:10.3f} | {target_mean:10.3f} | '
                          f'{self.lambda_H1.item():7.4f} | {self.lambda_H2.item():6.3f} | '
                          f'{self.lambda_H3.item():6.3f} | {self.lambda_H4.item():6.2f} | '
                          f'{lr_current:8.1e}')

                scheduler_hydrogen.step()

            except Exception as e:
                print(f"氢气约束计算失败 (epoch {epoch}): {e}")
                # 如果计算失败，跳过这次迭代
                continue

        print(f'氢气参数训练完成，最终Loss: {hydrogen_loss.item():.3e}')
        print()

    def predict(self, X, x_scal):
        x = X[:, 0:].to(device)

        # 注意：这里不要强制设置为 eval 模式，让调用者决定
        # self.dnn.eval()  # 注释掉这行
        u, log_var = self.net_u(x)
        f, *_ = self.net_f_V(X, x_scal)
        u = u.detach().cpu().numpy()
        log_var = log_var.detach().cpu().numpy()
        return u, log_var


def get_MC_samples(network, X, x_scal, mc_times=64, dropout=0.6):
    """
    蒙特卡洛采样函数 - 通过dropout计算认知不确定性

    Parameters:
    - network: 神经网络模型
    - X: 输入数据
    - x_scal: 输入数据的scaler
    - mc_times: 蒙特卡洛采样次数
    - dropout_multiplier: dropout率的倍数（如2.0表示将原dropout率乘以2）
    """
    pred_v_dropout = []
    pred_v_no_dropout = []
    a_u = []

    # 保存原始的dropout率
    original_dropout_rates = {}

    # 收集并保存所有dropout层的原始丢弃率
    for name, module in network.dnn.named_modules():
        if isinstance(module, torch.nn.Dropout):
            original_dropout_rates[name] = module.p

    print(f"开始MC采样，采样次数: {mc_times}，dropout: {dropout}")
    print("原始dropout率:")
    for name, rate in original_dropout_rates.items():
        print(f"  {name}: {rate:.3f}")

    # 第一阶段：禁用 dropout 获取基准预测
    network.dnn.eval()  # 禁用 dropout
    for t in range(mc_times):
        prediction, var = network.predict(X, x_scal)
        pred_v_no_dropout.append(prediction)

    # 修改dropout率
    print("\n调整后的dropout率:")
    for name, module in network.dnn.named_modules():
        if isinstance(module, torch.nn.Dropout):
            # 增加dropout率，但不超过0.9
            new_rate = dropout
            module.p = new_rate
            print(f"  {name}: {original_dropout_rates[name]:.3f} -> {new_rate:.3f}")

    # 第二阶段：开启 dropout 计算认知不确定性
    for t in range(mc_times):
        if t % 200 == 0:
            print(f"MC采样进度: {t}/{mc_times}")

        network.dnn.train()  # 开启训练模式以激活 dropout
        prediction, var = network.predict(X, x_scal)
        pred_v_dropout.append(prediction)
        a_u.append(var)

    # 恢复原始的dropout率
    print("\n恢复原始dropout率...")
    for name, module in network.dnn.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = original_dropout_rates[name]

    # 恢复评估模式
    network.dnn.eval()

    pred_v_dropout = np.array(pred_v_dropout)
    pred_v_no_dropout = np.array(pred_v_no_dropout)
    a_u = np.array(a_u)

    # 使用 dropout 的预测作为最终预测
    pred_mean = np.mean(pred_v_no_dropout, axis=0)

    # 偶然不确定性：来自模型输出的方差
    a_u = np.sqrt(np.exp(np.mean(a_u, axis=0)))

    # 认知不确定性：来自dropout导致的预测变化
    e_u = np.sqrt(np.var(pred_v_dropout, axis=0))

    # 输出统计信息
    print(f"\nMC采样完成:")

    return pred_mean.squeeze(), a_u.squeeze(), e_u.squeeze()


def view_uncertainty_only(fig_title, pred_val, ale_val, epi_val, n_test, window_size=50, data_info=None,
                          # 你可在调用时调整以下字号
                          title_size=16, label_size=12, tick_size=10, legend_size=10,
                          # 正常区均值虚线的外观
                          normal_mean_color='purple', normal_mean_linestyle='--', normal_mean_linewidth=1.5):
    """
    仅展示不确定性分布的可视化（英文图文，拆两张图）：
    图A: Aleatoric uncertainty（单子图）
    图B: Epistemic uncertainty（单子图 + 正常状态均值虚线）
    - 增加：计算正常数据的平均认知不确定度，并在图中绘制虚线基准
    - 原先两子图 -> 两张图；宽度不变，高度为原来一半
    代码注释保持中文
    """
    x_test_indices = np.arange(n_test)

    # 认知不确定性做移动平均（保留原逻辑）
    def moving_average_improved(data, window):
        if len(data) < window:
            return data
        try:
            import pandas as pd
            return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
        except ImportError:
            filtered = np.zeros_like(data)
            half = window // 2
            for i in range(len(data)):
                s = max(0, i - half)
                e = min(len(data), i + half + 1)
                filtered[i] = np.mean(data[s:e])
            return filtered

    epi_val_filtered = moving_average_improved(epi_val, window_size)

    print("Epistemic uncertainty filtering info:")
    print(f"  window: {window_size}")
    print(f"  raw range: [{np.min(epi_val):.6f}, {np.max(epi_val):.6f}]")
    print(f"  filtered range: [{np.min(epi_val_filtered):.6f}, {np.max(epi_val_filtered):.6f}]")

    # ===== 计算“正常状态”的认知不确定度均值 =====
    # 依据 data_info['boundary_lines'][0] 划分正常段；若不可得，则用整个序列均值
    normal_mean = None
    normal_end = None
    if data_info and 'boundary_lines' in data_info and len(data_info['boundary_lines']) > 0:
        normal_end = data_info['boundary_lines'][0]
        normal_end = min(int(normal_end), n_test)
        if normal_end > 0:
            normal_mean = float(np.mean(epi_val[:normal_end]))
        else:
            normal_mean = float(np.mean(epi_val))
    else:
        normal_mean = float(np.mean(epi_val))

    # ========== 图A：Aleatoric uncertainty（单图，高度减半） ==========
    figA, axA = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
    axA.scatter(x_test_indices, ale_val, c="blue", s=7, label="Aleatoric uncertainty", alpha=0.7)
    axA.set_ylabel('Uncertainty magnitude', fontsize=label_size)
    axA.set_title('Aleatoric uncertainty distribution', fontsize=title_size)
    axA.legend(fontsize=legend_size)
    axA.grid(True, alpha=0.3)
    axA.tick_params(axis='both', labelsize=tick_size)

    # 分界线和英文区域标注
    if data_info and 'boundary_lines' in data_info:
        for boundary in data_info['boundary_lines'][:-1]:
            axA.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # 标注 Normal
        if normal_end is not None and normal_end > 0:
            y_pos = axA.get_ylim()[1] * 0.95
            axA.text(normal_end / 2, y_pos, 'Normal',
                     ha='center', fontsize=max(9, tick_size),
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))

        # 标注各 Fault
        if 'fault_data_list' in data_info:
            y_pos2 = axA.get_ylim()[1] * 0.85
            for i_fault, (_, _, _label_cn) in enumerate(data_info['fault_data_list']):
                start_pos = data_info['boundary_lines'][i_fault]
                end_pos = data_info['boundary_lines'][i_fault + 1]
                mid_pos = (start_pos + end_pos) / 2
                axA.text(mid_pos, y_pos2, f'Fault-{i_fault + 1}',
                         ha='center', fontsize=max(8, tick_size - 1),
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.7))

    figA.suptitle(fig_title, fontsize=title_size)
    plt.tight_layout()
    plt.show()

    # ========== 图B：Epistemic uncertainty（单图 + 正常均值虚线） ==========
    figB, axB = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
    # axB.scatter(x_test_indices, epi_val, c="lightcoral", s=5, label="Epistemic uncertainty (raw)", alpha=0.5)
    axB.plot(x_test_indices, epi_val_filtered, c="red", linewidth=4,
             label=f"Epistemic uncertainty (filtered, window={window_size})", alpha=0.8)
    for side in ["left", "right", "top", "bottom"]:
        axB.spines[side].set_linewidth(2.5)  # 边框粗细
    # 正常状态均值虚线
    if normal_mean is not None and np.isfinite(normal_mean):
        axB.axhline(y=normal_mean, color=normal_mean_color, linestyle=normal_mean_linestyle,
                    linewidth=normal_mean_linewidth, label=f'Normal mean = {normal_mean:.4f}')

    axB.set_xlabel('Sample index', fontsize=label_size)
    axB.set_ylabel('Epistemic uncertainty', fontsize=label_size)
    # axB.set_title('Epistemic uncertainty distribution', fontsize=title_size)
    # axB.legend(fontsize=legend_size)
    # axB.grid(True, alpha=0.3)
    axB.tick_params(axis='both', labelsize=tick_size)

    # 分界线和英文区域标注
    if data_info and 'boundary_lines' in data_info:
        for boundary in data_info['boundary_lines'][:-1]:
            axB.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        #
        # if normal_end is not None and normal_end > 0:
        #     y_top = axB.get_ylim()[1]
        #     axB.text(normal_end / 2, y_top * 0.95, 'Normal',
        #              ha='center', fontsize=max(9, tick_size),
        #              bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
        #
        # if 'fault_data_list' in data_info:
        #     y_top = axB.get_ylim()[1]
        #     for i_fault, (_, _, _label_cn) in enumerate(data_info['fault_data_list']):
        #         start_pos = data_info['boundary_lines'][i_fault]
        #         end_pos = data_info['boundary_lines'][i_fault + 1]
        #         mid_pos = (start_pos + end_pos) / 2
        #         axB.text(mid_pos, y_top * 0.85, f'Fault-{i_fault + 1}',
        #                  ha='center', fontsize=max(8, tick_size - 1),
        #                  bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.7))

    # figB.suptitle(fig_title, fontsize=title_size)
    plt.tight_layout()
    plt.show()

def plot_model_results_detailed_split(model, dataset, data_info=None, fig_title="Detailed Model Analysis",
                                      windows=100,
                                      # 你可在调用时调整以下字号
                                      title_size=16, label_size=26, tick_size=10, legend_size=10):
    """
    单图 + 双y轴版本：
      - 左y轴：Voltage (真实 vs 预测)
      - 右y轴：Model residual（残差），通过调整y轴范围让残差整体位于图下方
      - 仿照 view_uncertainty_only 中 Epistemic 图的风格
      - 加上 boundary_lines 竖虚线划分 Normal / Fault 区域
      - 本版本额外改动：
         1. 去掉标题
         2. 右侧 y 轴轴线 & 标签改为绿色
         3. 三条曲线进一步加粗
         4. 图例移到右侧中部，减少遮挡
         5. 增加残差 = 0 的水平虚线
    """

    # 解包数据集
    x_train, y_train, x_test, y_test, scaler_X, scaler_Y, data_info_dict = dataset
    if data_info is None:
        data_info = data_info_dict

    # 数据准备（反归一化）
    x_rescal = scaler_X.inverse_transform(x_test.detach().cpu().numpy())
    y_rescal = scaler_Y.inverse_transform(y_test.detach().cpu().numpy()).flatten()

    # 使用数据点序号作为x轴
    n_samples = len(y_test)
    x_indices = np.arange(n_samples)

    # 计算预测与物理约束
    model.dnn.eval()
    f_pred, V_act, V_ohmic, V_conc, E_nerst, V_out_est, i, il, V_out = model.net_f_V(x_test, scaler_X)
    u_pred, log_var = model.predict(x_test, scaler_X)
    u_pred = scaler_Y.inverse_transform(u_pred).flatten()

    # 残差
    voltage_error = y_rescal - u_pred

    # ====================== 物理项（统计用） ======================
    f_pred_cpu = f_pred.detach().cpu().numpy().flatten()

    # 温度、氢、氧（下方统计需要）
    f_T_residual, T_out_predicted, T_out_real = model.net_f_T(x_test, scaler_X)
    f_T_residual_cpu = f_T_residual.detach().cpu().numpy().flatten()

    f_H_residual, actual_ratio_H, target_ratio_H, *_ = model.net_f_H(x_test, scaler_X)
    f_H_residual_cpu = f_H_residual.detach().cpu().numpy().flatten()
    actual_ratio_H_cpu = actual_ratio_H.detach().cpu().numpy().flatten()
    target_ratio_H_cpu = target_ratio_H.detach().cpu().numpy().flatten()

    f_O_residual, actual_ratio_O, target_ratio_O, *_ = model.net_f_O(x_test, scaler_X)
    f_O_residual_cpu = f_O_residual.detach().cpu().numpy().flatten()
    actual_ratio_O_cpu = actual_ratio_O.detach().cpu().numpy().flatten()
    target_ratio_O_cpu = target_ratio_O.detach().cpu().numpy().flatten()

    # ====================== 绘图：单图 + 双y轴 ======================
    fig, ax_left = plt.subplots(1, 1, figsize=(14, 6))

    # 电压：左 y 轴（曲线更粗）
    ln1 = ax_left.plot(
        x_indices, y_rescal,
        'b-', linewidth=3.5,
        label='Measured voltage', alpha=0.9
    )
    ln2 = ax_left.plot(
        x_indices, u_pred,
        'r--', linewidth=3.5,
        label='Model output', alpha=0.9
    )

    ax_left.set_ylabel('Voltage (V)', fontsize=label_size, color='black')
    ax_left.tick_params(axis='y', labelsize=tick_size)
    ax_left.tick_params(axis='x', labelsize=tick_size)
    ax_left.set_xlabel('Sample index', fontsize=label_size)

    # 设置左 y 轴范围（稍微向下扩展一倍，以便电压曲线整体偏上）
    v_min = min(y_rescal.min(), u_pred.min())
    v_max = max(y_rescal.max(), u_pred.max())
    v_margin = 0.05 * (v_max - v_min + 1e-6)
    ax_left.set_ylim(v_min - v_margin - (v_max - v_min), v_max + v_margin)

    # 残差：右 y 轴（曲线更粗，颜色绿色）
    ax_right = ax_left.twinx()
    ln3 = ax_right.plot(
        x_indices, voltage_error,
        color='green', linewidth=3.5,
        label='Model residual', alpha=0.9
    )

    # 右侧 y 轴轴线 & 标签设为绿色
    ax_right.set_ylabel('Model residual (V)', fontsize=label_size, color='green')
    ax_right.tick_params(axis='y', labelsize=tick_size, colors='green')
    ax_right.spines['right'].set_color('green')

    # ===== 调整右 y 轴范围，使残差整体“在下方” =====
    err_abs_max = np.max(np.abs(voltage_error)) + 1e-6
    # 让右轴下界为 -1.2 * max_abs_error，上界为 0.2 * max_abs_error
    ax_right.set_ylim(-1.2 * err_abs_max, 3.2 * err_abs_max)

    # ===== 残差=0 的虚线（绿色，方便观察变化） =====
    ax_right.axhline(
        y=0.0, color='green', linestyle='--',
        linewidth=2.0, alpha=0.7
    )

    # ===== 仿照 Epistemic 图的风格：边框变粗 =====
    for side in ["left", "right", "top", "bottom"]:
        ax_left.spines[side].set_linewidth(2.5)
        ax_right.spines[side].set_linewidth(2.5)

    # ===== boundary_lines 竖虚线（Normal / Fault 区域）=====
    if data_info and 'boundary_lines' in data_info:
        boundary_lines = data_info['boundary_lines']
        for boundary in boundary_lines[:-1]:
            ax_left.axvline(
                x=boundary,
                color='gray', linestyle='--',
                alpha=0.5, linewidth=2
            )

    # ===== 合并图例，放到右侧中部，避免遮挡主要波形 =====
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    # loc='center right'：右侧中间；bbox_to_anchor 可微调
    ax_left.legend(
        lines, labels,
        fontsize=legend_size,
        loc='center left'
    )

    # 1. 删除标题 —— 不再调用 set_title，或确保为空
    # ax_left.set_title('')

    plt.tight_layout()
    plt.show()

    # =================== 统计信息（终端打印，保持原有结构） ===================
    voltage_mae = np.mean(np.abs(voltage_error))
    voltage_rmse = np.sqrt(np.mean(voltage_error ** 2))
    voltage_r2 = 1 - np.sum(voltage_error ** 2) / np.sum((y_rescal - np.mean(y_rescal)) ** 2)

    physics_v_mae = np.mean(np.abs(f_pred_cpu))
    physics_v_rmse = np.sqrt(np.mean(f_pred_cpu ** 2))

    temp_mae_orig = np.mean(np.abs(f_T_residual_cpu))
    temp_rmse_orig = np.sqrt(np.mean(f_T_residual_cpu ** 2))

    # 平滑函数（仅用于统计，不再绘图）
    def _moving_avg_simple(x, w=windows):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode='same')

    f_T_smooth = _moving_avg_simple(f_T_residual_cpu, windows)
    temp_mae_smooth = np.mean(np.abs(f_T_smooth))
    temp_rmse_smooth = np.sqrt(np.mean(f_T_smooth ** 2))

    hydrogen_mae_orig = np.mean(np.abs(f_H_residual_cpu))
    hydrogen_rmse_orig = np.sqrt(np.mean(f_H_residual_cpu ** 2))
    oxygen_mae_orig = np.mean(np.abs(f_O_residual_cpu))
    oxygen_rmse_orig = np.sqrt(np.mean(f_O_residual_cpu ** 2))

    print("=" * 60)
    print("Model prediction statistics")
    print("=" * 60)
    print(f"Voltage:")
    print(f"  MAE: {voltage_mae:.6f} V")
    print(f"  RMSE: {voltage_rmse:.6f} V")
    print(f"  R^2: {voltage_r2:.6f}")
    print(f"  Max abs error: {np.max(np.abs(voltage_error)):.6f} V")
    print()
    print(f"Voltage physics consistency:")
    print(f"  Residual MAE: {physics_v_mae:.6f}")
    print(f"  Residual RMSE: {physics_v_rmse:.6f}")
    print()
    print(f"Temperature physics consistency:")
    print(f"  Original MAE: {temp_mae_orig:.6f} °C → Smoothed MAE: {temp_mae_smooth:.6f} °C")
    print(f"  Original RMSE: {temp_rmse_orig:.6f} °C → Smoothed RMSE: {temp_rmse_smooth:.6f} °C")
    print()
    print(f"Hydrogen physics consistency:")
    print(f"  Residual MAE: {hydrogen_mae_orig:.6f}")
    print(f"  Residual RMSE: {hydrogen_rmse_orig:.6f}")
    print(f"  Actual ratio range: [{actual_ratio_H_cpu.min():.3f}, {actual_ratio_H_cpu.max():.3f}]")
    print(f"  Target ratio range: [{target_ratio_H_cpu.min():.3f}, {target_ratio_H_cpu.max():.3f}]")
    print()
    print(f"Oxygen physics consistency:")
    print(f"  Residual MAE: {oxygen_mae_orig:.6f}")
    print(f"  Residual RMSE: {oxygen_rmse_orig:.6f}")
    print(f"  Actual ratio range: [{actual_ratio_O_cpu.min():.3f}, {actual_ratio_O_cpu.max():.3f}]")
    print(f"  Target ratio range: [{target_ratio_O_cpu.min():.3f}, {target_ratio_O_cpu.max():.3f}]")
    print("=" * 60)

    return {
        'voltage_mae': voltage_mae,
        'voltage_rmse': voltage_rmse,
        'voltage_r2': voltage_r2,
        'physics_v_mae': physics_v_mae,
        'temp_mae_smooth': temp_mae_smooth,
        'hydrogen_mae': hydrogen_mae_orig,
        'oxygen_mae': oxygen_mae_orig
    }

def _moving_average_centered(arr: np.ndarray, window: int) -> np.ndarray:
    """居中滑动平均，min_periods=1；优先用pandas，若不可用则退回numpy实现。"""
    try:
        import pandas as pd
        return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values
    except Exception:
        n = len(arr)
        if n == 0:
            return arr
        out = np.empty(n, dtype=float)
        half = window // 2
        for i in range(n):
            s = max(0, i - half)
            e = min(n, i + half + 1)
            out[i] = arr[s:e].mean()
        return out


def smooth_by_segments(values: np.ndarray, boundary_lines: list, window: int) -> np.ndarray:
    """
    按 boundary_lines 定义的段落分别做居中滑动平均，避免跨段“信息泄漏”。
    - boundary_lines 约定：依次给出每个段的“结束位置（exclusive）”，
      如 [normal_end, normal_end+f1, ..., total_samples]。
    """
    values = np.asarray(values, dtype=float).copy()
    n = len(values)
    out = np.empty_like(values, dtype=float)

    # 若未提供或不合法，则退化为整体平滑
    if not boundary_lines or boundary_lines[-1] != n:
        # 尝试容错：补齐最后边界
        if not boundary_lines or boundary_lines[-1] < n:
            return _moving_average_centered(values, window)
        # 若最后边界 > n，截断
        boundary_lines = [b for b in boundary_lines if 0 < b <= n]

    starts = [0] + boundary_lines[:-1]
    ends = boundary_lines

    for s, e in zip(starts, ends):
        seg = values[s:e]
        out[s:e] = _moving_average_centered(seg, window)
    return out


# 更简化和稳定的版本
# 更简化和稳定的版本
def create_comprehensive_results_array_v2(model, dataset, mc_times=2000, dropout=0.2):
    """
    创建包含所有结果的综合数组（22列）：
      0-7:  反归一化输入
      8:    真实输出(反归一化)
      9:    预测输出(反归一化)
      10:   偶然不确定性 ale（std，反归一化后）
      11:   认知不确定性 epi（std，反归一化后）
      12:   预测残差 y_true - y_pred（反归一化域）
      13:   电压物理残差
      14:   温度物理残差
      15:   氢气物理残差
      16:   氧气物理残差
      17:   标签（0=正常，其余为各故障段索引）
      18:   物理模型输出电压（堆栈电压，物理域）
      19:   物理模型输出温度 T_out_predicted（物理域）
      20:   氢气实际过量比 actual_excess_ratio_H
      21:   氧气实际过量比 actual_excess_ratio_O
    说明：
      - 不确定度统一由 get_MC_samples 计算（返回标准化域量纲），随后反归一化；
      - 不确定度做分段平滑（smooth_by_segments）。
    """
    # 解包数据集（兼容七元/九元）
    if len(dataset) == 9:
        x_train, y_train, x_val, y_val, x_test, y_test, scaler_X, scaler_Y, data_info = dataset
    else:
        x_train, y_train, x_test, y_test, scaler_X, scaler_Y, data_info = dataset

    print("正在计算综合结果数组（统一MC采样）...")

    # 反归一化输入与真实输出
    x_test_np = x_test.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()
    x_test_rescaled = scaler_X.inverse_transform(x_test_np)
    y_test_rescaled = scaler_Y.inverse_transform(y_test_np).flatten()

    # 1) 预测均值与不确定度（统一调用，当前返回为“标准化域”值）
    pred_mean_norm, ale_std_norm, epi_std_norm = get_MC_samples(
        model, x_test, scaler_X, mc_times=mc_times, dropout=dropout
    )

    # 将预测均值与不确定度反归一化到“物理域”
    # MinMaxScaler: y_norm = y * scale + min  => y = (y_norm - min) / scale
    lo_y = float(scaler_Y.feature_range[0])
    hi_y = float(scaler_Y.feature_range[1])
    data_min_y = scaler_Y.data_min_.astype(np.float64)  # shape (1,)
    data_max_y = scaler_Y.data_max_.astype(np.float64)  # shape (1,)
    scale_y = (hi_y - lo_y) / (data_max_y - data_min_y + 1e-12)  # shape (1,)
    min_y = lo_y - data_min_y * scale_y  # shape (1,)

    # 反归一化预测均值
    pred_mean_rescaled = (pred_mean_norm - min_y) / (scale_y + 1e-12)

    # 不确定度是标准差，线性缩放系数对 std 的反变换只需除以 scale（不加平移项）
    ale_std_rescaled = ale_std_norm / (scale_y + 1e-12)
    epi_std_rescaled = epi_std_norm / (scale_y + 1e-12)

    # squeeze 到 1D
    pred_mean_rescaled = np.asarray(pred_mean_rescaled).reshape(-1)
    ale_std_rescaled = np.asarray(ale_std_rescaled).reshape(-1)
    epi_std_rescaled = np.asarray(epi_std_rescaled).reshape(-1)

    # 预测残差（统一在反归一化域）
    prediction_residual = y_test_rescaled - pred_mean_rescaled

    # 2) 物理约束项 & 物理模型输出
    print("正在计算物理约束和物理模型输出...")
    model.dnn.eval()

    # 电压物理模型：同时拿残差和物理模型输出电压
    f_V_pred, V_act, V_ohmic, V_conc, E_nerst, V_out_est_stack, i, il, V_out_stack_true = model.net_f_V(x_test,
                                                                                                        scaler_X)
    # V_out_est_stack 是 net_f_V 中返回的 V_out_est*5，即堆栈电压估计
    V_out_phys_np = V_out_est_stack.detach().cpu().numpy().flatten()

    # 温度物理模型：拿残差和预测温度
    f_T_residual, T_out_predicted, T_out_real = model.net_f_T_simple(x_test, scaler_X)
    T_out_phys_np = T_out_predicted.detach().cpu().numpy().flatten()

    # 氢气模型：拿残差和实际过量比
    f_H_residual, actual_ratio_H, target_ratio_H, *_ = model.net_f_H(x_test, scaler_X)
    f_H_pred_np = f_H_residual.detach().cpu().numpy().flatten()
    actual_ratio_H_np = actual_ratio_H.detach().cpu().numpy().flatten()

    # 氧气模型：拿残差和实际过量比
    f_O_residual, actual_ratio_O, target_ratio_O, *_ = model.net_f_O(x_test, scaler_X)
    f_O_pred_np = f_O_residual.detach().cpu().numpy().flatten()
    actual_ratio_O_np = actual_ratio_O.detach().cpu().numpy().flatten()

    # 电压物理残差
    f_V_pred_np = f_V_pred.detach().cpu().numpy().flatten()
    # 温度物理残差
    f_T_pred_np = f_T_residual.detach().cpu().numpy().flatten()

    # 3) 分段平滑不确定度
    smooth_window = 200
    n_samples = len(x_test)
    boundaries = None
    if data_info and 'boundary_lines' in data_info and len(data_info['boundary_lines']) > 0:
        boundaries = data_info['boundary_lines']
        if boundaries[-1] != n_samples:
            boundaries = boundaries + [n_samples]

    if boundaries:
        ale_std_smooth = smooth_by_segments(ale_std_rescaled, boundaries, smooth_window)
        epi_std_smooth = smooth_by_segments(epi_std_rescaled, boundaries, smooth_window)
    else:
        ale_std_smooth = _moving_average_centered(ale_std_rescaled, smooth_window)
        epi_std_smooth = _moving_average_centered(epi_std_rescaled, smooth_window)

    # 4) 标签
    fault_labels = create_fault_labels(n_samples, data_info)

    # 5) 组装综合数组（22列）
    results_array = np.zeros((n_samples, 22), dtype=float)
    results_array[:, 0:8] = x_test_rescaled
    results_array[:, 8] = y_test_rescaled
    results_array[:, 9] = pred_mean_rescaled
    results_array[:, 10] = ale_std_smooth
    results_array[:, 11] = epi_std_smooth
    results_array[:, 12] = prediction_residual
    results_array[:, 13] = f_V_pred_np
    results_array[:, 14] = f_T_pred_np
    results_array[:, 15] = f_H_pred_np
    results_array[:, 16] = f_O_pred_np
    results_array[:, 17] = fault_labels
    # 新增的 4 列：物理模型输出
    results_array[:, 18] = V_out_phys_np  # 物理模型输出电压（堆栈电压）
    results_array[:, 19] = T_out_phys_np  # 物理模型输出温度
    results_array[:, 20] = actual_ratio_H_np  # 氢气实际过量比
    results_array[:, 21] = actual_ratio_O_np  # 氧气实际过量比

    print("综合结果数组创建完成！（不确定度已统一采样并平滑，量纲在物理域一致，新增了物理模型输出列）")
    return results_array


def create_fault_labels(n_samples, data_info):
    """
    创建故障标签数组，自动从data_info获取标签名称
    """

    fault_labels = np.zeros(n_samples)

    if data_info and 'boundary_lines' in data_info:
        # 正常数据区域 (标签为0)
        normal_end = data_info['boundary_lines'][0]
        fault_labels[:normal_end] = 0

        # 故障数据区域
        if 'fault_data_list' in data_info:
            for i, (_, _, label) in enumerate(data_info['fault_data_list']):
                start_idx = data_info['boundary_lines'][i]
                end_idx = data_info['boundary_lines'][i + 1]
                fault_labels[start_idx:end_idx] = i + 1  # 故障标签从1开始

                print(f"故障标签 {i + 1}: {label}, 范围: [{start_idx}:{end_idx - 1}]")

    print(f"故障标签统计:")
    unique_labels, counts = np.unique(fault_labels, return_counts=True)

    # 动态构建标签名称字典
    label_names = {0: "正常数据"}
    if data_info and 'fault_data_list' in data_info:
        for i, (_, _, label) in enumerate(data_info['fault_data_list']):
            label_names[i + 1] = label

    for label, count in zip(unique_labels, counts):
        label_name = label_names.get(int(label), f"未知故障{int(label)}")
        print(f"  标签 {int(label)} ({label_name}): {count} 个样本")

    return fault_labels


def mainmain(x):
    x = 1
    return x


if __name__ == "__main__":
    # 数据路径配置
    data_path_normal = (
        r'F:\2-同步库\3-内部保密资料\4-1-燃料电池测试数据\3-氢璞\qingpu_stack_test\Normal_Data_m\Data_m\Polar-1.mat')

    # 故障数据路径（使用新生成的文件）
    fault_data_folder = r'F:\2-同步库\3-内部保密资料\4-1-燃料电池测试数据\3-氢璞\整理后的故障数据\short'

    # 故障数据文件路径
    fault_data_paths = {
        # 水淹故障
        'water_flooding_108A': os.path.join(fault_data_folder, '水淹_108A.mat'),
        'water_flooding_270A': os.path.join(fault_data_folder, '水淹_270A.mat'),
        'water_flooding_405A': os.path.join(fault_data_folder, '水淹_405A.mat'),

        # 氧饥饿故障
        'oxygen_starvation_108A': os.path.join(fault_data_folder, '氧饥饿_108A.mat'),
        'oxygen_starvation_270A': os.path.join(fault_data_folder, '氧饥饿_270A.mat'),
        'oxygen_starvation_405A': os.path.join(fault_data_folder, '氧饥饿_405A.mat'),

        # 膜干故障
        'membrane_drying_108A': os.path.join(fault_data_folder, '膜干_108A.mat'),
        'membrane_drying_270A': os.path.join(fault_data_folder, '膜干_270A.mat'),
        'membrane_drying_405A': os.path.join(fault_data_folder, '膜干_405A.mat'),

        # 氢饥饿故障
        'hydrogen_starvation_108A': os.path.join(fault_data_folder, '氢饥饿_108A.mat'),
        'hydrogen_starvation_270A': os.path.join(fault_data_folder, '氢饥饿_270A.mat'),
        'hydrogen_starvation_405A': os.path.join(fault_data_folder, '氢饥饿_405A.mat'),
    }

    print("=" * 60)
    print("第一步：分别加载原始数据（不归一化）")
    print("=" * 60)

    # 加载正常数据
    X_normal, Y_normal = load_data_normal_raw(data_path_normal)

    # 检查并加载故障数据
    fault_data_list = []

    # 定义要加载的故障数据
    selected_faults = [
        ('water_flooding_108A', '水淹故障(108A)'),
        ('water_flooding_270A', '水淹故障(270A)'),
        ('water_flooding_405A', '水淹故障(405A)'),
        ('oxygen_starvation_108A', '氧饥饿故障(108A)'),
        ('oxygen_starvation_270A', '氧饥饿故障(270A)'),
        ('oxygen_starvation_405A', '氧饥饿故障(405A)'),
        ('membrane_drying_108A', '膜干故障(108A)'),
        ('membrane_drying_270A', '膜干故障(270A)'),
        ('membrane_drying_405A', '膜干故障(405A)'),
        ('hydrogen_starvation_108A', '氢饥饿故障(108A)'),
        ('hydrogen_starvation_270A', '氢饥饿故障(270A)'),
        ('hydrogen_starvation_405A', '氢饥饿故障(405A)'),
    ]

    # 加载选定的故障数据
    for fault_key, fault_label in selected_faults:
        fault_path = fault_data_paths[fault_key]

        if os.path.exists(fault_path):
            try:
                X_fault, Y_fault = load_data_fault_raw(fault_path)
                fault_data_list.append((X_fault, Y_fault, fault_label))
                print(f"✓ 成功加载: {fault_label}")
            except Exception as e:
                print(f"✗ 加载失败 {fault_label}: {e}")
        else:
            print(f"✗ 文件不存在: {fault_path}")

    print(f"\n总共加载了 {len(fault_data_list)} 个故障数据集")

    # 合并数据集，统一归一化，划分训练测试集
    Dataset = combine_and_normalize_datasets(
        normal_data=(X_normal, Y_normal),
        fault_data_list=fault_data_list,
        training_rate=1,  # 从正常数据中抽取100%作为训练集
        noise_config=None,  # 可选添加噪声
        seed=42
    )

    # %%
    # 设置DNN网络层数
    Layers = [Dataset[0].shape[1], 256, 256, 256, Dataset[1].shape[1]]  # 网络层数
    # 构建模型，输入中也包含了真实数据用于计算物理规律
    Model_Pinn = PhysicsInformedNN(*Dataset[0:2], Layers, Dataset[4], Dataset[5], p=0.2, logvar=True)  # 丢弃率
    # 第一阶段：训练DNN参数
    Model_Pinn.train_dnn(nIter=4001)
    # 第二阶段：训练lambda参数
    Model_Pinn.train_lambda(nIter=4001, dnn_para=False)
    Model_Pinn.train_lambda(nIter=4001, dnn_para=True)
    Model_Pinn.train_dnn(nIter=8001)
    # 第三阶段：训练温度模型参数
    Model_Pinn.train_thermal(nIter=10001)
    # 第四阶段：训练氢气模型参数（新增）
    Model_Pinn.train_hydrogen(nIter=8001)
    # 第五阶段：训练氧气模型参数（新增）
    Model_Pinn.train_oxygen(nIter=8001)

    # %% 创建综合结果数组（统一计算不确定度、测试集为全部数据）
    comprehensive_results = create_comprehensive_results_array_v2(
        Model_Pinn, Dataset, mc_times=2000, dropout=0.4
    )

    print(f"\n最终结果数组形状: {comprehensive_results.shape}")
    print("列结构说明:")
    print("  列0: 电流")
    print("  列1: 冷却水流量")
    print("  列2: 冷却水入堆温度")
    print("  列3: 氢气入堆压力")
    print("  列4: 空气入堆压力")
    print("  列5: 冷却水出堆温度 (电堆温度)")
    print("  列6: 氢气流量")
    print("  列7: 空气流量")
    print("  列8: 真实输出 (反归一化)")
    print("  列9: 预测输出 (反归一化)")
    print("  列10: 偶然不确定性")
    print("  列11: 认知不确定性")
    print("  列12: 预测残差")
    print("  列13: 电压约束惩罚")
    print("  列14: 温度约束惩罚")
    print("  列15: 氢气约束惩罚")
    print("  列16: 氧气约束惩罚")
    print("  列17: 故障标签")
    print("  列18: 物理模型输出电压（堆栈电压）")
    print("  列19: 物理模型输出温度")
    print("  列20: 氢气实际过量比")
    print("  列21: 氧气实际过量比")
    # 保存数据
    data = {'comprehensive_results': comprehensive_results}
    scipy.io.savemat('F01_output.mat', data)
    print("数组已成功保存为 F01_output.mat 文件")

    # %% 绘图
    pred_rescaled = comprehensive_results[:, 9]
    ale_rescaled = comprehensive_results[:, 10]
    epi_rescaled = comprehensive_results[:, 11]
    n_test = comprehensive_results.shape[0]

    plot_results = plot_model_results_detailed_split(
        Model_Pinn, Dataset, fig_title="Fuel cell voltage prediction",
        windows=100, title_size=24, label_size=24, tick_size=20, legend_size=18
    )

    view_uncertainty_only(
        "Uncertainty overview",
        pred_rescaled, ale_rescaled, epi_rescaled,
        n_test=n_test, window_size=200, data_info=Dataset[6],
        title_size=24, label_size=22, tick_size=16, legend_size=11,
        normal_mean_color='purple', normal_mean_linestyle='--', normal_mean_linewidth=3
    )
    # ===== 温度对比图：真实温度 vs 物理模型输出温度 =====
    true_temp = comprehensive_results[:, 5]  # 冷却水出堆温度 (真实)
    phys_temp = comprehensive_results[:, 19]  # 物理模型输出温度
    x_idx = np.arange(n_test)

    plt.figure(figsize=(14, 5))
    plt.plot(x_idx, true_temp, 'b-', linewidth=2, label='真实温度')
    plt.plot(x_idx, phys_temp, 'r--', linewidth=2, label='物理模型输出温度')
    plt.xlabel('样本索引', fontsize=22)
    plt.ylabel('温度 (°C)', fontsize=22)
    plt.title('真实温度 vs 物理模型输出温度', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=18)

    # 如果希望标出正常/故障分界线
    if Dataset[6] and 'boundary_lines' in Dataset[6]:
        for boundary in Dataset[6]['boundary_lines'][:-1]:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    plt.show()


