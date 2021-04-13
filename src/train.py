import torch
from torch import flatten
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from src.train_base import *


# training type 설정해서 -> Trainer 가져옴
def get_trainer(config,args, device, data_loader, log_writer, type):
    return Trainer(config, args, device, data_loader, log_writer, type=type)


def get_optimizer(model, config, args_optim):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=config.train.weight_decay)

    if args_optim == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=1e-05, weight_decay=config.train.weight_decay)

def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda = lambda epoch: 0.95**epoch, last_epoch=-1, verbose=False)

class Trainer:
    def __init__(self,config, args, device, data_loader, log_writer, type):
        self.config = config
        self.args = args
        self.pretrained_weights_path = config.data_info[args.dataset].weights
        
        self.device = device
        self.data_loader = data_loader
        self.log_writer = log_writer
        self.type = type

        self.loss_function = CrossEntropyLoss()
        self.global_step = 0

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_lr_schedule(self, scheduler):
        self.scheduler = scheduler

    def train_epoch(self, model, epoch, global_step=None):
        if self.type=="train":
            model.train()
        else:
            model.eval()

        model.to(self.device)
        loss_save = list()

        for data in tqdm(self.data_loader, desc='Epoch: {}'.format(epoch)):

            input_data = data[0].to(self.device)
            label = data[1].long()
            label = label.to(self.device)
            y = model.forward(input_data) # (batch_size, n_classes)
            loss = self.loss_function(y, label)

            if self.type =="train":
                self.optim_process(model, loss)
                self.global_step += 1
                self.write_tb(loss, global_step)
            else:
                loss_save.append(loss.item())

            self.evaluator = self.evaluate(y, label)

        if self.type != 'train':
            loss = sum(loss_save)/len(loss_save)
            self.write_tb(loss, global_step)
            return loss

    def optim_process(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
        self.optimizer.step()
        self.scheduler.step()

    def write_tb(self, loss, global_step):
        if self.type =='train':
            lr = self.optimizer.param_groups[0]["lr"]
            self.log_writer.add_scalar("train/loss", loss, global_step)
            self.log_writer.add_scalar("train/lr", lr, global_step)
            
        else:
            self.log_writer.add_scalar("valid/loss", loss, global_step)

    def evaluate(self, y, label):
        a = torch.argmax(F.log_softmax(y, dim=1), dim=1)
        accuracy = len(a[a == label])/len(a)*100
        return accuracy
        









def get_data_loader(config=None, dataset=None,data_type="train", shuffle=True,workers=10, drop_last=False):
    batch_size = config.train.batch_size
    dataset = Make_Dataset(config.path_preprocessed, dataset, data_type)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=drop_last,collate_fn = padded_seq)
    return data_loader

def padded_seq(samples):
    def padd(samples):
        length = [len(s) for s in samples]
        max_length = max(length)
        batch = torch.zeros(len(samples),max_length).to(torch.long)
        for idx, sample in enumerate(samples):
            batch[idx, :length[idx]] = torch.LongTensor(sample)
        return batch
    data = []
    label = []
    for sample in samples:
        data.append(sample["data"])
        label.append(sample["label"])
    data = padd(data)
    label = torch.Tensor(label)
    return data, label

class Make_Dataset(Dataset):
    def __init__(self, file_path, dataset, data_type):
        data = torch.load(file_path)
        self.dataset = data[dataset]
        
        if data_type =="train":
            self.data = self.dataset["train"]
            self.data_label = self.dataset["train_label"]
        elif data_type =="test":
            self.data = self.dataset["test"]
            self.data_label = self.dataset["test_label"]
        else:
            self.data = self.dataset["dev"]
            self.data_label = self.dataset["dev_label"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        ret_dict = dict()
        ret_dict["data"] = self.data[idx]
        ret_dict["label"] = self.data_label[idx]

        return ret_dict
