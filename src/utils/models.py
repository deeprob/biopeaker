#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F


###############
# classifiers #
###############

class TFPerceptron(nn.Module):
    
    def __init__(self, args):
        # dropout pron only added for consistency
        super(TFPerceptron, self).__init__()
        self.bn1 = nn.BatchNorm1d(args.feat_size)
        self.fc1 = nn.Linear(args.feat_size, 1)
        
    def forward(self, x_in):
        y_out = self.fc1(self.bn1(torch.flatten(x_in, start_dim=1)))
        return y_out


class TFMLP(nn.Module):
    
    def __init__(self, args):        
        
        super(TFMLP, self).__init__()
        self.dropout = args.dropout_prob

        self.bn1 = nn.BatchNorm1d(args.feat_size)
        self.fc1 = nn.Linear(args.feat_size, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn3 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 1)
        
    def forward(self, x_in):
        y_out = F.dropout(F.relu(self.bn2(self.fc1(self.bn1(torch.flatten(x_in, start_dim=1))))), p=self.dropout, training=self.training)
        y_out = F.dropout(F.relu(self.bn3(self.fc2(y_out))), p=self.dropout, training=self.training)  
        y_out = self.fc3(y_out)
        return y_out



############
# encoders #
############


class L1Block(nn.Module):

    def __init__(self):
        super(L1Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L2Block(nn.Module):

    def __init__(self):
        super(L2Block, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L3Block(nn.Module):

    def __init__(self):
        super(L3Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(200)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)

        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                   self.conv2, self.bn2, nn.ReLU(inplace=True),
                                   self.conv3, self.bn3)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L4Block(nn.Module):

    def __init__(self):
        super(L4Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn2 = nn.BatchNorm2d(200)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                   self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.prelayer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                      self.conv2, self.bn2, nn.ReLU(inplace=True))


        self.layer1 = nn.Sequential(*[L1Block() for x in range(2)])
        self.layer2 = nn.Sequential(*[L2Block() for x in range(2)])
        self.layer3 = nn.Sequential(*[L3Block() for x in range(2)])
        self.layer4 = nn.Sequential(*[L4Block() for x in range(2)])


        self.c1to2 = nn.Conv2d(64, 128, (3, 1), stride=(1, 1), padding=(1, 0))
        self.b1to2 = nn.BatchNorm2d(128)
        self.l1tol2 = nn.Sequential(self.c1to2, self.b1to2,nn.ReLU(inplace=True))

        self.c2to3 = nn.Conv2d(128, 200, (1, 1), padding=(3, 0))
        self.b2to3 = nn.BatchNorm2d(200)
        self.l2tol3 = nn.Sequential(self.c2to3, self.b2to3,nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(2200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 1)
        self.flayer = self.final_layer()

    def final_layer(self):
        self.conv3 = nn.Conv2d(200, 200, (7,1), stride =(1,1), padding = (4,0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))


    def forward(self, x_in):
        x_in = x_in.view(-1, 4, 500, 1)  # batch_size x 4 x 500 x 1 [4 channels]

        out = self.prelayer(x_in)
        out = self.layer1(out)
        out = self.layer2(self.l1tol2(out))
        out = self.maxpool1(out)
        out = self.layer3(self.l2tol3(out))
        out = self.maxpool2(out)
        out = self.layer4(out)
        out = self.flayer(out)
        out = self.maxpool3(out)
        out = out.view(-1, 2200)
        return out
