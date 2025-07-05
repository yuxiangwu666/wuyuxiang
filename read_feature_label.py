import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, fldictpath, feature_path,label_path):
        self.feature_path = feature_path
        self.label_path = label_path
        self.feature_label_pairs = self.get_feature_label_pairs(fldictpath)

    def __len__(self):
        return len(self.feature_label_pairs)

    def __getitem__(self, idx):
        feature_name, label_name = self.feature_label_pairs[idx]
        feature = np.load(os.path.join(self.feature_path, feature_name))
        label = np.load(os.path.join(self.label_path, label_name))
        feature = np.transpose(feature, (2, 0, 1)).astype(np.float32)
        label = np.transpose(label, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(feature), torch.from_numpy(label)

    def get_feature_label_pairs(self, fldictpath):
        pairs = []
        with open(fldictpath, "r") as f:
            for line in f:
                feature_name, label_name = line.strip().split(",")
                feature_name = feature_name.strip("feature/")
                label_name = label_name.strip("label/")
                pairs.append((feature_name, label_name))
        return pairs

if __name__ == "__main__":# 使用示例
    fldict_path = "D:\\学习资料\\学习资料\\毕业论文\CircuitNet\\routability_ir_drop_prediction\\files\\train_N28.csv"
    feature_path = "D:\\py vs code\\gra_design\\training_set\\DRC\\feature"
    label_path = "D:\\py vs code\gra_design\\training_set\\DRC\\label"
    dataset = MyDataset(fldict_path,feature_path,label_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    featrue , label = next(iter(dataloader))
    print(featrue.shape, label.shape)# 使用DataLoader加载数据集
    # 迭代DataLoader以访问数据

