import sys, os
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
sys.path.append(os.getcwd())

from src.utils import *
from src.train_base import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode")
parser.add_argument('--type', type=str, default="rand")
parser.add_argument('--dataset', type=str, default="CR")

parser.add_argument('--config', type=str, default='default')
parser.add_argument('--log', type=str, default="loss")
parser.add_argument('--gpu', type=str, default=None)

# training
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--optim', type=str, default="adam")
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--L2s', type=int, default=5) # need to change default same as paper's
parser.add_argument('--use_earlystop', type=int, default=0)

parser.add_argument("--total_steps", type=int, default=26000)
parser.add_argument("--eval_period", type=int, default=400) 

#hyperparams
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--folds', type=int, default=10)

args = parser.parse_args()
print("train model : {}".format(args.type))

assert args.type in ["rand", 'static', "non-static", "multi-channel"]
assert args.dataset in ["CR", "MR", "SST1", 'SST2', "Subj", 'TREC', "MPQA"]
assert args.optim in ["adam","adagrad"]

config = load_config(args.config)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

oj = os.path.join


from src.model import CNN_Classifier as classifier
import src.train as train

# loss graph 
trainer, evaluator = None, None
tb_loc = oj(args.log, "tb")
chkpnt_loc = oj(args.log, "chkpnt")

if not os.path.exists(args.log):
    os.mkdir(args.log)
    os.mkdir(tb_loc)
    os.mkdir(chkpnt_loc)

writer = SummaryWriter(tb_loc)

# data load
train_loader = train.get_data_loader(config, args.dataset, "train") # data, data_label 둘다 가져옴
dev_loader = train.get_data_loader(config,args.dataset, "dev")
test_loader = train.get_data_loader(config,args.dataset,"test")
print("dataset iteration num: train {} | dev {} | test {} ".format(len(train_loader), len(dev_loader), len(test_loader)))

# model load
model = classifier(config=config, args=args)

for name, param in model.named_parameters():
    print(name, ":", param.requires_grad)

# trainer load
trainer = train.get_trainer(config, args, device, train_loader, writer, type='train')
dev_trainer = train.get_trainer(config, args, device, dev_loader, writer, type="valid")
test_trainer = train.get_trainer(config, args, device, test_loader, writer, type="test")

optimizer = train.get_optimizer(model, config, args.optim)
scheduler = train.get_lr_scheduler(optimizer)

trainer.init_optimizer(optimizer)
trainer.init_lr_schedule(scheduler)

total_steps = int(args.total_steps / len(train_loader))+1
eval_epoch = max(int(args.eval_period / len(train_loader))+1, 0)
print("Total_steps : {}".format(total_steps))

early_stop_loss = []
for epoch in range(1, args.epochs+1):
    trainer.train_epoch(model, epoch)
    valid_loss = dev_trainer.train_epoch(model, epoch, trainer.global_step)
    early_stop_loss.append(valid_loss)
    if args.use_earlystop and early_stop_loss[-2] < early_stop_loss[-1]:
        break

    if epoch % eval_epoch == 0:
        torch.save({
            'epoch':epoch,
             'model_state_dict':model.state_dict()},
            os.path.join(chkpnt_loc, 'model.{}.ckpt'.format(epoch)))

        train_accuracy = trainer.evaluator
        valid_accuracy = dev_trainer.evaluator
        print("epoch : {} | train_accuracy : {} | valid_accuracy : {} ".format(epoch, train_accuracy, valid_accuracy))

    





