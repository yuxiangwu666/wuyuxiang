import re
import matplotlib.pyplot as plt

def draw_loss(file_path, threshold=0.01):
    # 正则表达式模式
    iter_pattern = r"Iter:\s*([\d]+)"
    loss_pattern = r"Loss:\s*([\d\.]+)"

    # 存储提取的迭代次数和损失值
    iters = []
    losses = []

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 查找迭代次数
            iter_match = re.search(iter_pattern, line)
            if iter_match:
                iters.append(int(iter_match.group(1)))
            
            # 查找损失值
            loss_match = re.search(loss_pattern, line)
            if loss_match and len(iters) > len(losses):
                losses.append(float(loss_match.group(1)))

    # 打印提取的迭代次数和损失值
    print(f'Iterations: {iters}')
    print(f'Losses: {losses}')

    # 过滤掉差距过大的损失值
    filtered_iters = []
    filtered_losses = []
    for i in range(len(losses)):
        if losses[i] <= threshold:
            filtered_iters.append(iters[i])
            filtered_losses.append(losses[i])

    # 找到最低点
    if filtered_losses:
        min_loss = min(filtered_losses)
        min_loss_index = filtered_losses.index(min_loss)
        min_loss_iter = filtered_iters[min_loss_index]

        # 打印结果
        plt.plot(filtered_iters, filtered_losses, label='Loss')
        
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss vs. Iterations")
        

        # 标记最低点
        plt.scatter(min_loss_iter, min_loss, color='red', label=f'Min Loss: {min_loss:.4f}')
        plt.legend()
        plt.show()
    else:
        print("No losses below the threshold were found.")

draw_loss("D:\\py vs code\\gra_design\\train_congestion.out")