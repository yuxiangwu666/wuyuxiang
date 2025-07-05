import read_feature_label as rfl
import os
from tqdm import tqdm
from my_model import attention_unet
from gpdl_model import GPDL
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from routenet import RouteNet
import csv
from atten_routenet import atten_RouteNet
def calculate_nrmse(prediction, label):
    mse = mean_squared_error(label, prediction)
    rmse = np.sqrt(mse)
    nrmse = rmse / (label.max() - label.min())
    return nrmse

def test_model(param_path, fldict_path, feature_path, label_path):
    print(f'==> Testing model: {param_path}')
    dataset = rfl.MyDataset(fldict_path, feature_path, label_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = attention_unet()
    model = model.cuda()
    
    # 加载模型参数
    checkpoint = torch.load(param_path)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    total_loss = 0.0
    losses = []
    nrmse_values = []
    ssim_values = []
    num_batches = len(dataloader)
    
    with torch.no_grad():  # 关闭梯度计算
        with tqdm(total=num_batches) as bar:
            for feature, label in dataloader:
                feature, label = feature.cuda(), label.cuda()
                feature = feature.float()
                label = label.float()
                prediction = model(feature)
                loss = nn.MSELoss()(prediction, label)
                
                total_loss += loss.item()
                losses.append(loss.item())
                
                # 将预测值和标签从 GPU 移动到 CPU，并转换为 numpy 数组
                prediction_np = prediction.cpu().numpy().squeeze()
                label_np = label.cpu().numpy().squeeze()
                
                # 计算 NRMSE
                nrmse = calculate_nrmse(prediction_np, label_np)
                nrmse_values.append(nrmse)
                
                # 计算 SSIM
                ssim_value = ssim(prediction_np, label_np, data_range=label_np.max() - label_np.min())
                ssim_values.append(ssim_value)
                
                bar.update(1)
    
    average_loss = total_loss / num_batches
    loss_variance = np.var(losses)
    average_nrmse = np.mean(nrmse_values)
    average_ssim = np.mean(ssim_values)
    
    print(f'Average MSE Loss: {average_loss:.4f}')
    print(f'Loss Variance: {loss_variance:.4f}')
    print(f'Average NRMSE: {average_nrmse:.4f}')
    print(f'Average SSIM: {average_ssim:.4f}')
    
    return average_loss, loss_variance, average_nrmse, average_ssim

def test_all_models(param_dir, fldict_path, feature_path, label_path, output_file):
    # 确保输出文件所在的文件夹存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件夹中所有模型参数文件
    model_files = [os.path.join(param_dir, f) for f in os.listdir(param_dir) if f.endswith('.pth')]
    
    # 打开 CSV 文件，写入测试结果
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Average MSE Loss', 'Loss Variance', 'Average NRMSE', 'Average SSIM'])
        
        for model_file in model_files:
            average_loss, loss_variance, average_nrmse, average_ssim = test_model(
                model_file, fldict_path, feature_path, label_path
            )
            writer.writerow([model_file, average_loss, loss_variance, average_nrmse, average_ssim])

if __name__ == '__main__':
    param_dir = "D:\\py vs code\\gra_design\\atten_model_three\\DRC"
    fldict_path = "D:\\学习资料\\学习资料\\毕业论文\\CircuitNet\\routability_ir_drop_prediction\\files\\test_N28.csv"
    feature_path = "D:\\py vs code\\gra_design\\training_set\\DRC\\feature"
    label_path = "D:\\py vs code\\gra_design\\training_set\\DRC\\label"
    output_file = "D:\\py vs code\\gra_design\\test_result\\drc_atten_one\\test_results1.csv"
    
    test_all_models(param_dir, fldict_path, feature_path, label_path, output_file)