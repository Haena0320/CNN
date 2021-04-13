from tqdm import tqdm
import torch
from torch import flatten
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
import os

def get_data_loader(config=None, dataset=None, shuffle=True, data_type="train"):
    batch_size = config.train.batch_size
    file_path = os.path.join(config.path_preprocessed+".pkl")
    dataset = Make_Dataset(file_path, dataset, data_type)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class Make_Dataset(Dataset):
    def __init__(self, file_path, dataset, data_type):
        with open(file_path, 'rb') as f:
            data = torch.load(f)

        dataset = data[dataset]
        if data_type =="train":
            self.data = dataset["train"]
            self.data_label = dataset["train_label"]
        elif data_type =="test":
            self.data = dataset["test"]
            self.data_label = dataset["test_label"]
        else:
            self.data = dataset["val"]
            self.data_label = dataset["val_label"]

    def __len__(self):
        return len(self.data)

    def __getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.data_label[idx]







    





