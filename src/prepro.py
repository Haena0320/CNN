import re
import numpy as np
import torch
import re

# 데이티 전처리 추가 - 저자 전처리 코드 참고
def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_vocab(data_list=None, dataset="", padding=1):
    max_sent_len = 0
    word2id = dict()
    word2id["<padding>"] = 0
    id = 1  # padding id = 0, word ids start after padding id
    for data in data_list:
        if dataset in ["CR", "SST1", "SST2", 'MPQA']:
            f = open(data, encoding="utf-8")
        else:  # MR, SUBJ, TREC
            f = open(data, encoding='ISO-8859-1')
        for line in f:
            if dataset in ["SST1", "SST2"]:
                line = clean_str_sst(line)

            elif dataset in ["CR", "MPQA", "MR", "SUBJ"]:
                line = clean_str(line)
            else: # TREC
                line = clean_str(line, True)

            words = line.split()
            max_sent_len = max(max_sent_len, len(words))
            for word in words:
                if word not in word2id:
                    word2id[word] = id
                    id += 1
        f.close()

    print("word embeddings : {} ".format(len(word2id.keys()) - 2))
    print('max_sentence length : {}'.format(max_sent_len))
    return max_sent_len, word2id


def load_data(dataset, train_name, test_name="", dev_name="", padding=4):
    f_names = [train_name]
    if not test_name == '': f_names.append(test_name)
    if not dev_name == "": f_names.append(dev_name)
    max_sent_len, word2id = get_vocab(f_names, dataset, padding)

    # data load
    train = []
    test = []
    dev = []

    train_label = []
    test_label = []
    dev_label = []

    # all data (train, test, dev)
    files = []
    data = []
    data_label = []

    f_train = open(train_name, 'rb')
    files.append(f_train)
    data.append(train)
    data_label.append(train_label)

    if not test_name == '':
        f_test = open(test_name, 'rb')
        files.append(f_test)
        data.append(test)
        data_label.append(test_label)

    if not dev_name == '':
        f_dev = open(dev_name, "rb")
        files.append(f_dev)
        data.append(dev)
        data_label.append(dev_label)

    for d, lb, f in zip(data, data_label, files):
        while True:
            if dataset in ["CR", "SST1", "SST2", 'MPQA']:
                line = f.readline()
                if not line:
                    break

                y = int(line[0])
                words = line[1:]
                sent = []

                for word in words:
                    sent.append(word)

            else:
                line = f.readline().decode('ISO-8859-1')
                if dataset =="TREC":
                    line = clean_str(line, True)
                else:
                    line = clean_str(line)
                line = line.split()
                if not line:
                    break

                y = int(line[0])
                words = line[1:]
                sent = []

                for word in words:
                    sent.append(word2id[word])

            # end padding
            if len(sent) < max_sent_len + padding:
                sent.extend([0] * (max_sent_len + padding - len(sent)))
            # start padding
            sent = [0] * padding + sent

            d.append(sent)
            lb.append(y)
    f_train.close()
    if not test_name == "":
        f_test.close()
    if not dev_name == "":
        f_dev.close()

    train_label = label_adjust(train_label)
    if not test_name =="":
        test_label = label_adjust(test_label)
    if not dev_name =="":
        dev_label = label_adjust(dev_label)
    print(train)
    return word2id, np.array(train, dtype=object), np.array(train_label), np.array(test), np.array(test_label), np.array(dev), np.array(dev_label)


def label_adjust(label):
    mini = min(list(set(label)))
    return [i - mini for i in label]


def adjust_data(train, train_label, test, test_label, dev, dev_label):
    limit = len(train) // 10
    if (len(test) ==0 & len(dev) ==0):

        test = train[:limit]
        test_label = train_label[:limit]

        dev = train[limit:2*limit]
        dev_label = train_label[limit:2*limit]
        
    elif (len(test) ==0 & len(dev) !=0):
        test = train[:limit]
        test_label = train_label[:limit]

    elif (len(test) != 0 & len(dev)==0):
        dev = train[:limit]
        dev_label = train_label[:limit]

    else:
        pass

    return train, train_label, test, test_label, dev, dev_label



def pretrained_weights(model, word2id, save_path):
    embedding_dim= 300
    vocab_size = len(word2id.items())
    weights = np.zeros((vocab_size, embedding_dim))
    cnt = 0
    for word, id in word2id.items():
        try:
            wv = model[word]
            weights[id] = wv
            cnt+=1
        except:
            weights[id] = np.random.randn(embedding_dim)*0.2 # max_norm 0.2

    print("total vocabulary :{} | pretrained weights vocabulary : {} | random weights vocabulary : {}".format(vocab_size, cnt, vocab_size-cnt))
    torch.save(weights, save_path)
    return None