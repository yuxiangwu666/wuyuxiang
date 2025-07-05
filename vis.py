import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from routenet import RouteNet
import read_feature_label as rfl
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # 添加进度条
from my_model import attention_unet
from gpdl_model import GPDL
from atten_routenet import atten_RouteNet
from sklearn.preprocessing import MinMaxScaler

def calculate_nrmse(prediction, label):
    """
    计算 NRMSE（归一化均方根误差）。
    """
    mse = mean_squared_error(label, prediction)
    rmse = np.sqrt(mse)
    nrmse = rmse / (label.max() - label.min())
    return nrmse

def normalize_data(data):
    """
    将数据归一化到 [0, 1] 范围。
    """
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def threshold_data(data, threshold):
    """
    对数据进行阈值处理，低于阈值的部分设置为 0。
    """
    data_thresholded = np.where(data > threshold, data, 0)
    return data_thresholded

def test_and_find_closest_sample(model, dataloader, metric='ssim', condition='above'):
    """
    测试模型并找到最接近平均值的样本（根据指定的条件）。
    :param model: PyTorch 模型。
    :param dataloader: 数据加载器。
    :param metric: 指定使用的指标（'ssim' 或 'nrmse'）。
    :param condition: 筛选条件（'above' 表示高于平均值，'below' 表示低于平均值）。
    :return: 最接近样本的预测值、标签值、NRMSE 和 SSIM。
    """
    assert metric in ['ssim', 'nrmse'], "Metric must be 'ssim' or 'nrmse'."
    assert condition in ['above', 'below'], "Condition must be 'above' or 'below'."

    model.eval()
    nrmse_values = []
    predictions = []
    labels = []
    ssim_values = []

    with torch.no_grad():
        for feature, label in tqdm(dataloader, desc="Testing", unit="batch"):  # 添加进度条
            feature, label = feature.cuda(), label.cuda()
            feature = feature.float()
            label = label.float()
            prediction = model(feature)
            
            # 将预测值和标签值从 GPU 移动到 CPU，并转换为 numpy 数组
            prediction_np = prediction.cpu().numpy().squeeze()
            label_np = label.cpu().numpy().squeeze()
            
            # 计算 NRMSE 和 SSIM
            nrmse_value = calculate_nrmse(prediction_np, label_np)
            ssim_value = ssim(prediction_np, label_np, data_range=label_np.max() - label_np.min())
            
            # 保存结果
            nrmse_values.append(nrmse_value)
            ssim_values.append(ssim_value)
            predictions.append(prediction_np)
            labels.append(label_np)

    # 选择指标
    if metric == 'ssim':
        values = ssim_values
    else:
        values = nrmse_values

    # 计算平均值
    mean_value = np.mean(values)

    # 筛选出符合条件的样本
    if condition == 'above':
        indices = [i for i, val in enumerate(values) if val > mean_value]
    else:
        indices = [i for i, val in enumerate(values) if val < mean_value]

    if not indices:
        raise ValueError(f"No samples have {metric.upper()} {condition} the mean value.")

    # 找到符合条件中最接近平均值的样本
    closest_index = min(indices, key=lambda i: abs(values[i] - mean_value))
    closest_prediction = predictions[closest_index]
    closest_label = labels[closest_index]
    closest_nrmse = nrmse_values[closest_index]
    closest_ssim = ssim_values[closest_index]

    return closest_prediction, closest_label, closest_nrmse, closest_ssim

def visualize_prediction_and_label(prediction, label, nrmse_value, ssim_value, out_path):
    """
    可视化预测值和标签值，并保存为图片文件，同时标注 NRMSE 和 SSIM。
    """
    # 确保输出文件夹存在
    output_dir = os.path.dirname(out_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # 标签值
    plt.subplot(1, 2, 1)
    plt.imshow(label, cmap='plasma', vmin=label.min(), vmax=label.max())  # 使用高对比度的颜色映射
    plt.colorbar()
    plt.title('Ground Truth')

    # 预测值
    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap='plasma', vmin=label.min(), vmax=label.max())  # 保持与标签一致的范围
    plt.colorbar()
    plt.title(f'Prediction\nNRMSE: {nrmse_value:.4f}, SSIM: {ssim_value:.4f}')

    # 保存图片
    plt.savefig(out_path)
    plt.close()
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    # 模型参数文件路径
    param_dir = "D:\\py vs code\\gra_design\\atten_routenet_model_two\\DRC\\model_180000.pth"
    fldict_path = "D:\\学习资料\\学习资料\\毕业论文\\CircuitNet\\routability_ir_drop_prediction\\files\\test_N28.csv"
    feature_path = "D:\\py vs code\\gra_design\\training_set\\DRC\\feature"
    label_path = "D:\\py vs code\\gra_design\\training_set\\DRC\\label"

    # 输出图片路径
    out_path_above_mean_ssim = "D:\\py vs code\\gra_design\\test_result\\vis\\drc_atten_fcn\\closest_to_mean_visualization_above_mean_ssim.png"
    out_path_below_mean_nrmse = "D:\\py vs code\\gra_design\\test_result\\vis\\drc_atten_fcn\\closest_to_mean_visualization_below_mean_nrmse.png"

    # 加载数据集
    dataset = rfl.MyDataset(fldict_path, feature_path, label_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = atten_RouteNet()
    model = model.cuda()
    checkpoint = torch.load(param_dir)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 测试模型并找到最接近高于平均 SSIM 的样本
    closest_prediction_above_mean_ssim, closest_label_above_mean_ssim, closest_nrmse_above_mean_ssim, closest_ssim_above_mean_ssim = test_and_find_closest_sample(
        model, dataloader, metric='ssim', condition='above'
    )
    # 可视化最接近高于平均 SSIM 的样本
    visualize_prediction_and_label(
        closest_prediction_above_mean_ssim, closest_label_above_mean_ssim,
        closest_nrmse_above_mean_ssim, closest_ssim_above_mean_ssim,
        out_path_above_mean_ssim
    )

    # 测试模型并找到最接近低于平均 NRMSE 的样本
    closest_prediction_below_mean_nrmse, closest_label_below_mean_nrmse, closest_nrmse_below_mean_nrmse, closest_ssim_below_mean_nrmse = test_and_find_closest_sample(
        model, dataloader, metric='nrmse', condition='below'
    )
    # 可视化最接近低于平均 NRMSE 的样本
    visualize_prediction_and_label(
        closest_prediction_below_mean_nrmse, closest_label_below_mean_nrmse,
        closest_nrmse_below_mean_nrmse, closest_ssim_below_mean_nrmse,
        out_path_below_mean_nrmse
    )