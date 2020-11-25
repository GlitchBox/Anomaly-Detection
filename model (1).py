from __future__ import print_function
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
#from torchvision import datasets, transforms
import pickle
import pandas as pd
from time import time


class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()

    self.num_features = 49
    self.D = self.num_features * 10
    self.L = 64
    self.K = 1

    # self.conv_ = nn.Conv1d(400, 1, kernel_size=3, groups=1, padding=1)

    """self.feature_extractor_part1 = nn.Sequential(
        nn.Conv1d(10, 10, kernel_size=3, groups=1, padding=1),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Conv1d(10, 10, kernel_size=3, groups=1, padding=1),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Conv1d(10, 10, kernel_size=3, groups=1, padding=1),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Conv1d(10, 8, kernel_size=3, groups=1, padding=1),
        nn.Dropout(0.1),
        nn.ReLU()
        # nn.Conv1d(8, 6, kernel_size=3, groups=1, padding=1),
        # nn.ReLU(),
        # nn.Conv1d(6, 4, kernel_size=3, groups=1, padding=1),
#             nn.ReLU(),
# #             nn.MaxPool2d(2, stride=2),
#             nn.Conv1d(4, 2, kernel_size=3, groups=1, padding=1),
#             nn.ReLU(),
# #             nn.MaxPool2d(2, stride=2),
#             nn.Conv1d(2, 1, kernel_size=3, groups=1, padding=1),
#             nn.ReLU(),
    )"""

    #         self.bn1 = nn.BatchNorm1d(self.D)

    #         self.bn2 = nn.BatchNorm1d(self.L)

    self.feature_extractor_part2 = nn.Sequential(
      nn.Linear(self.num_features * 10, self.num_features * 6),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.Linear(self.num_features * 6, self.num_features * 4),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.Linear(self.num_features * 4, self.L),
      nn.Dropout(0.5),
      nn.ReLU(),
    )

    self.lstm_feature_extractor = nn.LSTM(input_size=self.num_features,
                                          hidden_size=self.num_features,
                                          num_layers=2,
                                          batch_first=True,
                                          dropout=0.3,
                                          bidirectional=True)
    self.lstm_activation = nn.ReLU()

    self.attention = nn.Sequential(
      nn.Linear(self.L, self.L),
      nn.Dropout(0.1),
      nn.Tanh()
      #             nn.Linear(self.D, self.K)
    )

    self.gate = nn.Sequential(
      nn.Linear(self.L, self.L),
      nn.Dropout(0.1),
      nn.Sigmoid()
    )

    self.apply_gate = nn.Sequential(
      nn.Linear(self.L, self.K),
      nn.Dropout(0.1)
    )

    self.classifier = nn.Sequential(
      nn.Linear(self.L * self.K, 1),
      nn.Dropout(0.2),
      nn.Sigmoid()
    )

  def forward(self, x):
    # print('x.shape:',x.shape)
    x = x.squeeze(0)
    # print('x sequeezed shape:',x.shape)
    #         x = x.permute(0, 2, 1).contiguous()
    #         x = self.bn1(x)
    #         x = x.permute(0, 2, 1).contiguous()

    #         print('x.shape:',x.shape, x.device)

    # y = torch.zeros([x.shape[0], x.shape[1], self.D]).cuda()
    # for i in range(x.shape[0]):
    #     y[i, :, :].copy_(self.feature_extractor_part1(x[i]).squeeze().view(x.shape[1], self.D))

    # y = self.feature_extractor_part1(x).squeeze().view(x.shape[0], self.D)
    # print('y.shape:',y.shape, y.device)


    temp, _ = self.lstm_feature_extractor(x)
    temp = temp[:, :, :x.shape[2]] + temp[:, :, x.shape[2]:]
    temp = self.lstm_activation(temp)
    # print("lstm>>", temp.shape)

    # print('temp.shape:', temp.shape, temp.device)

    x = self.feature_extractor_part2(temp.squeeze().reshape(-1, temp.shape[1] * temp.shape[2]))
    # print('x.shape:',x.shape)

    #         x = x.permute(0, 2, 1).contiguous()
    #         x = self.bn2(x)
    #         x = x.permute(0, 2, 1).contiguous()
    #         print('x.shape:',x.shape)

    # =============LSTM=====================
    #         x, _ = self.lstm_feature_extractor(x)
    #         x = x[:, :, :self.L] + x[:, :, self.L:]
    #         x = self.lstm_activation(x)
    #         print("lstm>>", x.shape)
    # ====================================
    #         print('x.shape:',x.shape)
    #         H = self.feature_extractor_part1(x)
    #         H = H.view(-1, 50 * 4 * 4)
    #         H = self.feature_extractor_part2(H)  # NxL

    # ==============Attention===================================
    A = self.attention(x)  # NxK
    # print('A.shape (attention matrix shape):', A.shape)
    G = self.gate(x)  ######################
    #         print('G.shape (gating matrix shape):', G.shape)
    A = torch.mul(A, G)  #####################
    #         print('torch.mul(A, G) shape:', A.shape)
    A = self.apply_gate(A)
    # print('self.apply_gate(A) shape:', A.shape)
    A = torch.transpose(A, -1, -2)  # KxN
    # print('A.shape after transpose:', A.shape)
    A = F.softmax(A, dim=-1)  # softmax over N

    # ===============Apply-Attention===================================
    M = torch.matmul(A, x)  # KxL
    #         print('M.shape (torch.mm(A, x)):', M.shape, A.shape, x.shape)

    Y_prob = self.classifier(M)
    #         print('Y_prob.shape :', Y_prob.shape)
    Y_prob = Y_prob.squeeze(-1)
    #         print('Y_prob.shape :', Y_prob.shape)
    Y_hat = torch.ge(Y_prob, 0.5).float()
    #         print('Y_hat.shape,  Y_hat.shape :', Y_hat.shape, Y_hat.shape)

    return Y_prob, Y_hat, A

    # AUXILIARY METHODS

  def predict(self, X):
    Y_prob, Y_hat, Attention_weights = self.forward(X)
    # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    Y_prob = (Variable(Y_prob).data).cpu().numpy()
    Y_hat = (Variable(Y_hat).data).cpu().numpy()
    Attention_weights = (Variable(Attention_weights).data).cpu().numpy()

    return Y_prob, Y_hat, Attention_weights

  def calculate_classification_error(self, X, Y):
    Y = Y.float()
    _, Y_hat, _ = self.forward(X)
    error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

    return error, Y_hat

  def calculate_objective(self, X, Y):
    Y = Y.float()
    Y_prob, _, A = self.forward(X)

    # print('Y, Y_prob:', Y, Y_prob)

    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    #         print('neg_log_likelihood:', neg_log_likelihood.shape, Y_prob.shape)

    return neg_log_likelihood, A
