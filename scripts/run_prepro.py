import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
import gensim
import gensim.downloader as api
from src.utils import *
from src.prepro import *

import operator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="prepro0")
#parser.add_argument('--dataset', help='Data set', type=str, default="CR")
parser.add_argument('--word2vec', help='word2vec file', type=str)
parser.add_argument('--padding', help="padding around each sentence", type=int, default=4)
parser.add_argument('--config', type=str, default="default")

args = parser.parse_args()
config = load_config(args.config)

model = api.load("word2vec-google-news-300")

data = dict()
for dataset in ["SST1","SST2", "MR", "SUBJ", "CR", "MPQA", "TREC"]:
    print(dataset)
    data[dataset] = dict()
    train_path, dev_path, test_path = config.path_rawdata[dataset]
    # load data
    word2id, train, train_label, test, test_label, dev, dev_label = load_data(dataset, train_path, test_path, dev_path, padding=args.padding)
    # adjust dataset
    train, train_label, test, test_label, dev, dev_label = adjust_data(train, train_label, test, test_label, dev, dev_label)

    print("train {}, train_label {}".format(len(train), len(train_label)))
    print("test {} test_label {}".format(len(test), len(test_label)))
    print("dev {} dev_label {}".format(len(dev), len(dev_label)))

    data[dataset]["train"] = train
    data[dataset]["train_label"] = train_label 
    data[dataset]["test"] = test
    data[dataset]["test_label"] = test_label
    data[dataset]["dev"] = dev
    data[dataset]["dev_label"] = dev_label
    # word2id save
    torch.save(word2id, config.data_info[dataset].word2id)
    # pretrained weights save
    pretrained_weights(model, word2id, config.data_info[dataset].weights)

# dataset save
torch.save(data,config.path_preprocessed)
print("Done!")