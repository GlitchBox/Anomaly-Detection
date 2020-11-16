import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.embeddingDimension = 120
        self.K = 1

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
    #this is a trial attempt at building a feature extractor for the optical flow from i3d
    # input dimension will be (7,7,1024) or (m,7,7,1024) maybe?
    #output will be a 120 dimension vector for now
    def feature_extractor_opticalflow_i3d(self, opticalFlow, ifPool=False):
            
        #reshaping the opticalFlow, so that it is in channel-first order (m,1024,7,7)
        opticalFlow = opticalFlow.permute(0,3,1,2)
        if ifPool==True:
            opticalFlow = nn.MaxPool2d(kernel_size=3, stride=1)(opticalFlow)
            opticalFlow = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)
            opticalFlow = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)

        else:
            opticalFlow = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)
            
            opticalFlow = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)
            
        opticalFlow = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)(opticalFlow)
        opticalFlow = nn.ReLU()(opticalFlow)
            
        opticalFlow = opticalFlow.reshape(-1, 64*3*3)
        opticalFlow = nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension)
        opticalFlow = nn.ReLU()(opticalFlow)

        
        return opticalFlow

    #this is a trial attempt at building a feature extractor for the rgb output from i3d
    # input dimension will be (7,7,1024) or (m,7,7,1024) maybe?
    #output will be a 120 dimension vector for now
    def feature_extractor_rgb_i3d(self, rgb, ifPool=False):
            
        #reshaping the rgb input, so that it is in channel-first order (m,1024,7,7)
        rgb = rgb.permute(0,3,1,2)
        if ifPool==True:
            rgb = nn.MaxPool2d(kernel_size=3, stride=1)(rgb)
            rgb = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1)(rgb)
            rgb = nn.ReLU()(rgb)
            rgb = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)(rgb)
            rgb = nn.ReLU()(rgb)

        else:
            rgb = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1)(rgb)
            rgb = nn.ReLU()(rgb)
            
            rgb = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)(rgb)
            rgb = nn.ReLU()(rgb)
            
        rgb = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)(rgb)
        rgb = nn.ReLU()(rgb)
            
        rgb = rgb.reshape(-1, 64*3*3)
        rgb = nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension)
        rgb = nn.ReLU()(rgb)

        
        return rgb

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.embeddingDimension = 120
        self.K = 1

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
    
    #this is a trial attempt at building a feature extractor for the optical flow from i3d
    # input dimension will be (7,7,1024) or (m,7,7,1024) maybe?
    #output will be a 120 dimension vector for now
    def feature_extractor_opticalflow_i3d(self, opticalFlow, ifPool=False):
            
        #reshaping the opticalFlow, so that it is in channel-first order (m,1024,7,7)
        opticalFlow = opticalFlow.permute(0,3,1,2)
        if ifPool==True:
            opticalFlow = nn.MaxPool2d(kernel_size=3, stride=1)(opticalFlow)
            opticalFlow = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)
            opticalFlow = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)

        else:
            opticalFlow = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)
            
            opticalFlow = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)(opticalFlow)
            opticalFlow = nn.ReLU()(opticalFlow)
            
        opticalFlow = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)(opticalFlow)
        opticalFlow = nn.ReLU()(opticalFlow)
            
        opticalFlow = opticalFlow.reshape(-1, 64*3*3)
        opticalFlow = nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension)
        opticalFlow = nn.ReLU()(opticalFlow)

        
        return opticalFlow

    #this is a trial attempt at building a feature extractor for the rgb output from i3d
    # input dimension will be (7,7,1024) or (m,7,7,1024) maybe?
    #output will be a 120 dimension vector for now
    def feature_extractor_rgb_i3d(self, rgb, ifPool=False):
            
        #reshaping the rgb input, so that it is in channel-first order (m,1024,7,7)
        rgb = rgb.permute(0,3,1,2)
        if ifPool==True:
            rgb = nn.MaxPool2d(kernel_size=3, stride=1)(rgb)
            rgb = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1)(rgb)
            rgb = nn.ReLU()(rgb)
            rgb = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)(rgb)
            rgb = nn.ReLU()(rgb)

        else:
            rgb = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1)(rgb)
            rgb = nn.ReLU()(rgb)
            
            rgb = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)(rgb)
            rgb = nn.ReLU()(rgb)
            
        rgb = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)(rgb)
        rgb = nn.ReLU()(rgb)
            
        rgb = rgb.reshape(-1, 64*3*3)
        rgb = nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension)
        rgb = nn.ReLU()(rgb)

        
        return rgb
        
    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
