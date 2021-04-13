import torch
import torch.nn as nn
import numpy as np

class CNN_Classifier(nn.Module):
    def __init__(self, config, args, use_batch_norm=False, dropout_p =.5, window_sizes= [3,4,5], n_filters = [100,100,100]):

        data_info = config.data_info[args.dataset]
        model_info = config.model_info[args.type]

        self.vocab_size = data_info.vocab_size
        self.dimension = config.train.dimension
        self.n_classes = data_info.classes
        self.pretrained_weights_path = data_info.weights

        self.type = args.type
        self.pretrained = model_info.pretrained
        self.channel = int(model_info.channel)
        self.fine_tunned = model_info.fine_tunned

        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        self.window_sizes = window_sizes
        self.n_filters = n_filters
        super().__init__()
        if self.type == "rand":
            self.emb = nn.Embedding(self.vocab_size, self.dimension, max_norm=0.2)
        else:
            weights = torch.Tensor(torch.load(data_info.weights))
            if self.type =="static":
                self.emb = nn.Embedding.from_pretrained(weights)
            elif self.type =='non-static':
                self.emb = nn.Embedding.from_pretrained(weights, freeze=False)
            else: # multi-channel
                self.emb_1 = nn.Embedding.from_pretrained(weights,freeze=False)
                self.emb_2 = nn.Embedding.from_pretrained(weights)

        self.feature_extractors = nn.ModuleList()

        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
            nn.Sequential(nn.Conv2d(in_channels=1,  # 1 or 2 embedding layer
                                    out_channels=n_filter, # 100
                                    kernel_size=(window_size, self.dimension), # filter size
                                    stride=1,
                                    padding=(5,0)), # start, end padding
                          nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(self.dropout_p),
                          nn.Tanh()))

        self.fc = nn.Linear(sum(n_filters), self.n_classes)

    def forward(self, x): # x : ( batch_size, sentence_length)
        if self.type != "multi-channel":
            x = self.emb(x)
        else:
            x_1 = self.emb_1(x)
            x_2 = self.emb_2(x)
            x = (x_1+x_2)/2

        x = x.unsqueeze(1)  # x : ( batch_size, 1, length, embedding_dim)

        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(x) # cnn_out : (batch_size, n_filter, length-window_size+1, 1)
            cnn_out = nn.functional.max_pool1d(
                input=cnn_out.squeeze(-1),  # cnn_out : ( batch_size, n_filter, length-window_size+1 )
                kernel_size = cnn_out.size(-2)
            ).squeeze(-1)
            cnn_outs += [cnn_out]

        cnn_outs = torch.cat(cnn_outs, dim=-1) # cnn_outs : ( batch_size, n_filters (300)) 
        y = self.fc(cnn_outs) # y : (batch_size, n_classes)
        return y