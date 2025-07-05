import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read_feature_label import MyDataset 
from my_model import attention_unet
from routenet import RouteNet
from gpdl_model import GPDL
import os
from tqdm import tqdm
from atten_routenet import atten_RouteNet
def train(fldict_path, feature_path, label_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('==> Loading data')
    dataset = MyDataset(fldict_path, feature_path, label_path)
    dataloader=DataLoader(dataset, batch_size=8, shuffle=True)
    print('==> Building model')
    model = attention_unet()
    model = model.cuda()
    max_iters = 20000
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epoch_loss = 0
    print_freq = 100
    save_freq = 1000
    iter_num = 0
    while iter_num < max_iters:
        with tqdm(total=print_freq) as bar:
            for feature, label in dataloader:
                feature = feature.cuda()
                label = label.cuda()
                prediction = model(feature)
                optimizer.zero_grad()
                pixel_loss = criterion(prediction, label)
                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()
                iter_num += 1
                bar.update(1)
                if iter_num % print_freq == 0:
                    break
        print(f'Iter[{iter_num}]({iter_num}/({max_iters}):  Loss: {epoch_loss/print_freq}')
                    
        if iter_num % save_freq == 0:
            checkpoint(model,iter_num,save_path)
        epoch_loss = 0
            

def checkpoint(model,iters,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)
        
    model_out_path= os.path.join(save_path,f"model_{iters}.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    fldict_path = "D:\\学习资料\\学习资料\\毕业论文\CircuitNet\\routability_ir_drop_prediction\\files\\train_N28.csv"
    feature_path = "D:\\py vs code\\gra_design\\training_set\\congestion\\feature"
    label_path = "D:\\py vs code\gra_design\\training_set\\congestion\\label"
    save_path = "D:\\py vs code\\gra_design\\atten_model\\congestion"
    train(fldict_path, feature_path, label_path, save_path)
   