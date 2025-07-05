import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_number_from_model(model_name):
    """
    从模型名称中提取数字部分。
    假设模型名称格式为 'model_iters_XXXX.pth'，提取 XXXX。
    """
    match = re.search(r'\d+', model_name)
    return int(match.group()) if match else None

def read_metrics_from_csv(csv_file):
    """
    从 CSV 文件中读取模型编号、NRMSE 和 SSIM 数据。
    """
    models = []
    nrmse_values = []
    ssim_values = []

    # 读取 CSV 文件
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            model_name = row['Model']
            nrmse = float(row['Average NRMSE'])
            ssim = float(row['Average SSIM'])

            # 提取模型名称中的数字
            model_number = extract_number_from_model(model_name)
            if model_number is not None:
                models.append(model_number)
                nrmse_values.append(nrmse)
                ssim_values.append(ssim)

    # 按模型编号排序
    sorted_indices = sorted(range(len(models)), key=lambda i: models[i])
    models = [models[i] for i in sorted_indices]
    nrmse_values = [nrmse_values[i] for i in sorted_indices]
    ssim_values = [ssim_values[i] for i in sorted_indices]

    return models, nrmse_values, ssim_values

def calculate_stable_metrics(models, values, stable_ratio=0.7):
    """
    计算稳定区间的平均值和方差。
    :param models: 模型编号列表。
    :param values: 对应的指标值列表（如 NRMSE 或 SSIM）。
    :param stable_ratio: 稳定区间的比例（默认取最后 10% 的数据）。
    :return: 稳定区间的平均值和方差。
    """
    # 确定稳定区间的起始索引
    stable_start_index = int(len(models) * (1 - stable_ratio))
    stable_values = values[stable_start_index:]

    # 计算平均值和方差
    mean_value = np.mean(stable_values)
    variance_value = np.var(stable_values)

    return mean_value, variance_value

def plot_combined_metrics(csv_file1, csv_file2, csv_file3, output_dir):
    """
    从三个 CSV 文件中读取数据，并绘制 NRMSE 和 SSIM 的折线图，同时计算稳定区间的平均值和方差。
    """
    # 从三个文件中读取数据
    models1, nrmse_values1, ssim_values1 = read_metrics_from_csv(csv_file1)
    models2, nrmse_values2, ssim_values2 = read_metrics_from_csv(csv_file2)
    models3, nrmse_values3, ssim_values3 = read_metrics_from_csv(csv_file3)

    # 计算稳定区间的平均值和方差
    stable_ratio = 0.1  # 使用最后 10% 的数据作为稳定区间
    nrmse_mean1, nrmse_var1 = calculate_stable_metrics(models1, nrmse_values1, stable_ratio)
    ssim_mean1, ssim_var1 = calculate_stable_metrics(models1, ssim_values1, stable_ratio)
    nrmse_mean2, nrmse_var2 = calculate_stable_metrics(models2, nrmse_values2, stable_ratio)
    ssim_mean2, ssim_var2 = calculate_stable_metrics(models2, ssim_values2, stable_ratio)
    nrmse_mean3, nrmse_var3 = calculate_stable_metrics(models3, nrmse_values3, stable_ratio)
    ssim_mean3, ssim_var3 = calculate_stable_metrics(models3, ssim_values3, stable_ratio)

    # 打印稳定区间的结果
    print("Stable Metrics (Last 10% of Iterations):")
    print(f"Model 1 - NRMSE: Mean={nrmse_mean1:.4f}, Variance={nrmse_var1:.4f}")
    print(f"Model 1 - SSIM: Mean={ssim_mean1:.4f}, Variance={ssim_var1:.4f}")
    print(f"Model 2 - NRMSE: Mean={nrmse_mean2:.4f}, Variance={nrmse_var2:.4f}")
    print(f"Model 2 - SSIM: Mean={ssim_mean2:.4f}, Variance={ssim_var2:.4f}")
    print(f"Model 3 - NRMSE: Mean={nrmse_mean3:.4f}, Variance={nrmse_var3:.4f}")
    print(f"Model 3 - SSIM: Mean={ssim_mean3:.4f}, Variance={ssim_var3:.4f}")

    # 绘制 NRMSE 折线图
    plt.figure(figsize=(10, 6))
    plt.scatter(models1, nrmse_values1, label='atten_fcn NRMSE', color='blue', marker='o')
    plt.scatter(models2, nrmse_values2, label='fcn NRMSE', color='red', marker='x')
    plt.scatter(models3, nrmse_values3, label='atten_unet NRMSE', color='green', marker='s')

    # 曲线拟合
    z1 = np.polyfit(models1, nrmse_values1, 5)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(models2, nrmse_values2, 5)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(models3, nrmse_values3, 5)
    p3 = np.poly1d(z3)

    # 绘制拟合曲线
    x1 = np.linspace(min(models1), max(models1), 500)
    x2 = np.linspace(min(models2), max(models2), 500)
    x3 = np.linspace(min(models3), max(models3), 500)
    plt.plot(x1, p1(x1), color='blue', linestyle='--', label='Fitted Curve atten_fcn')
    plt.plot(x2, p2(x2), color='red', linestyle='--', label='Fitted Curve fcn')
    plt.plot(x3, p3(x3), color='green', linestyle='--', label='Fitted Curve atten_unet')

    plt.xlabel('Model Iterations')
    plt.ylabel('NRMSE')
    plt.title('NRMSE Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/nrmse_comparison.png')
    plt.close()

    # 绘制 SSIM 折线图
    plt.figure(figsize=(10, 6))
    plt.scatter(models1, ssim_values1, label='atten_fcn SSIM', color='blue', marker='o')
    plt.scatter(models2, ssim_values2, label='fcn SSIM', color='red', marker='x')
    plt.scatter(models3, ssim_values3, label='atten_unet SSIM', color='green', marker='s')

    # 曲线拟合
    z1 = np.polyfit(models1, ssim_values1, 5)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(models2, ssim_values2, 5)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(models3, ssim_values3, 5)
    p3 = np.poly1d(z3)

    # 绘制拟合曲线
    x1 = np.linspace(min(models1), max(models1), 500)
    x2 = np.linspace(min(models2), max(models2), 500)
    x3 = np.linspace(min(models3), max(models3), 500)
    plt.plot(x1, p1(x1), color='blue', linestyle='--', label='Fitted Curve atten_fcn')
    plt.plot(x2, p2(x2), color='red', linestyle='--', label='Fitted Curve fcn')
    plt.plot(x3, p3(x3), color='green', linestyle='--', label='Fitted Curve atten_unet')

    plt.xlabel('Model Iterations')
    plt.ylabel('SSIM')
    plt.title('SSIM Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/ssim_comparison.png')
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == '__main__':
    csv_file1 = "D:\\py vs code\\gra_design\\test_result\\congestion_atten_gpdl\\test_results2.csv"
    csv_file2 = "D:\\py vs code\\gra_design\\test_result\\congestion_gpdl\\test_results2.csv"
    csv_file3 = "D:\\py vs code\\gra_design\\test_result\\congestion_atten\\test_results2.csv"
    output_dir = "D:\\py vs code\\gra_design\\test_result\\congestion_three_compare\\plots_combined"

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plot_combined_metrics(csv_file1, csv_file2, csv_file3, output_dir)