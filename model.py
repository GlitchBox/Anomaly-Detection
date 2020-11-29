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
    def __init__(self, embeddingDimension=120):
        super(GatedAttention, self).__init__()
        # self.L = 500
        self.L = 256
        self.D = 64
        self.embeddingDimension = embeddingDimension #this could be a possible hyperparameter
        self.K = 1
        self.poolingPolicy = ["attention", "avg", "max"]

        self.i3d_opticalflow_extractor1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.i3d_opticalflow_extractor2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.i3d_opticalflow_extractor3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.ReLU()
        )

        self.i3d_opticalflow_extractor4 = nn.Sequential(
            nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension),
            nn.ReLU()
        )


        self.i3d_rgb_extractor1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.i3d_rgb_extractor2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.i3d_rgb_extractor3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.ReLU()
        )

        self.i3d_rgb_extractor4 = nn.Sequential(
            nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension),
            nn.ReLU()
        )

        self.openpose_dense1 = nn.Linear(in_features=25*3, out_features=self.embeddingDimension)

        self.attention_V_frame = nn.Sequential(
            nn.Linear(self.embeddingDimension, self.D),
            # nn.Linear(90, 60),
            nn.Tanh()
        )

        self.attention_U_frame = nn.Sequential(
            nn.Linear(self.embeddingDimension, self.D),
            # nn.Linear(90, 60),
            nn.Sigmoid()
        )

        self.attention_weights_frame = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
    
    #this is a trial attempt at building a feature extractor for the optical flow from i3d
    # input dimension will be (7,7,1024) or (m,7,7,1024) maybe?
    #I'm assuming, the input will be of the form ( 7, 7, 1024)
    #output will be a 120 dimension vector for now
    def feature_extractor_opticalflow_i3d(self, opticalFlow, ifPool=False):
            
        #reshaping the opticalFlow, so that it is in channel-first order (m,1024,7,7)
        # opticalFlow = opticalFlow.permute(0,3,1,2)
        
        #reshaping the opticalFlow, so that it is in channel-first order (1024,7,7)
        opticalFlow = opticalFlow.permute(2,0,1)
        opticalFlow = opticalFlow.unsqueeze(0) #including the batch size, the shape becomes (1,1024,7,7)
        if ifPool==True:
            opticalFlow = self.i3d_opticalflow_extractor1(opticalFlow)

        else:
            opticalFlow = self.i3d_opticalflow_extractor2(opticalFlow)
            
        opticalFlow = self.i3d_opticalflow_extractor3(opticalFlow)
        opticalFlow = opticalFlow.reshape(-1, 64*3*3)
        opticalFlow = self.i3d_opticalflow_extractor4(opticalFlow) #output shape (m, 120)

        return opticalFlow.squeeze(0) #output shape (120)
        # return opticalFlow #output shape (m,120)

    #this is a trial attempt at building a feature extractor for the rgb output from i3d
    # input dimension will be (7,7,1024) or (m,7,7,1024) maybe?
    #I'm assuming, the input will be of the form (7, 7, 1024)
    #output will be a 120 dimension vector for now
    def feature_extractor_rgb_i3d(self, rgb, ifPool=False):
            
        #reshaping the rgb input, so that it is in channel-first order (m,1024,7,7)
        rgb = rgb.permute(0,3,1,2)

        #reshaping the rgb input, so that it is in channel-first order (1024,7,7)
        rgb = rgb.permute(2,0,1)
        rgb = rgb.unsqueeze(0) #including the batch-size dimension, the shape becomes (1,1024,7,7)
        if ifPool==True:
            rgb = self.i3d_rgb_extractor1(rgb)

        else:
            rgb = self.i3d_rgb_extractor2(rgb)
            
        rgb = self.i3d_rgb_extractor3(rgb)
        rgb = rgb.reshape(-1, 64*3*3)
        rgb =  self.i3d_rgb_extractor4(rgb) #output shape (m, 120)

        return rgb.squeeze(0) #output shape (120)
        # return rgb #output shape (m,120)

    #the commented out lines are for when m == 1 and m is not included in the input shape
    def frame_encoder(self, openpose_instance_single_frame, pooling='attention'):
        #openpose_instance_single_frame will be of the size (m, human_count, 25, 3), here m is the number of people in a frame
        # human_count = openpose_instance_single_frame.shape[0]
        m = openpose_instance_single_frame.shape[0]
        human_count = openpose_instance_single_frame.shape[1]
        # H = openpose_instance_single_frame.reshape(human_count, 25*3)
        H = openpose_instance_single_frame.reshape(-1, human_count, 25*3)
        H = self.openpose_dense1(H) #output of this will be shape (m, human_count, 120)

        A = None
        if pooling ==  'attention' or pooling == 'max':
            A_V = self.attention_V_frame(H) #(m, human_count, 64)
            A_U = self.attention_U_frame(H) #(m, human_count, 64)
            A = self.attention_weights_frame(A_V*A_U) # (m, human_count, 1)
            # A = torch.transpose(A, 1, 0) #(1, human_count)
            A = A.permute(0, 2, 1) #(m, 1, human_count)
            
            if pooling == 'attention':
                # A = F.softmax(A, dim=1) # softmax over human_count, (1, human_count)
                A = F.softmax(A, dim=2) #softmax over human_count (m, 1, human_count), softmax doesn't have learnable parameters, hence it need not be declared in __init__
            else:
                A_ = torch.zeros((m, 1, human_count))
                # A_ = torch.zeros(( 1,human_count))
                # A_[0][A.argmax()] = 1
                for i in range(m):
                    A_[i][0][A[i][0].argmax()] = 1
                A = A_
        elif pooling == 'avg':
            A = torch.ones((m,1,human_count))/human_count
        
        M = torch.zeros((m,1,120))
        # M = torch.mm(A, H) #(1,120) 
        for i in range(m):
            M[i] = torch.mm(A[i], H[i]) #Shape of M (m,1,120)
        # return M.squeeze(0) #output shape (120)
        return M.squeeze(1) #output shape (m,120)

    #the commented out lines are for when m == 1 and m is not included in the input shape
    #I'm assuming that the single_instance will be "list" of tensors
    def openpose_instance_encoder(self, single_instance):
        encoded_frames = []
        instanceLen = len(single_instance)

        for i in range(instanceLen):
            encoded_frame = self.frame_encoder(single_instance[i])
            encoded_frame = nn.Linear(in_features=120, out_features=256)(encoded_frame) #output shape (1,256) or (m,256)
            # encoded_frame = encoded_frame.unsqueeze(0) #output shape (1,1,256)
            encoded_frame = encoded_frame.unsqueeze(1) #output shape (m,1,256)
            encoded_frames.append(encoded_frame)
        #now encoded_frames will be a list of tensors. The shape-> (10, 1, 1, 256) or (10, m , 1, 256)

        #now I'll prepare an LSTM cell
        input_dim = 256
        output_dim = 512
        layers_num = 1
        batch_size = encoded_frames[0].shape[0]
        
        lstm_layer = nn.LSTM(input_dim, output_dim, layers_num, batch_first=True)
        hidden_state = torch.randn(layers_num, batch_size, output_dim)
        cell_state = torch.randn(layers_num, batch_size, output_dim)
        hidden = (hidden_state, cell_state)
        
        for frame in encoded_frames:
            out, hidden = lstm_layer(frame, hidden)
        
        #now hidden[1] stores the cell state, ergo the long term memory
        #I'm using the cell state as the embedding
        outPutEmbedding = hidden[1] #shape (1,1,512) or (1,m,512)
        outPutEmbedding = nn.Linear(in_features=512, out_features=120)(outPutEmbedding) #shape (1,1,120) or (1,m,120)
        #return outPutEmbedding[0][0] #output shape (120)
        return outPutEmbedding[0] #output shape (m,120)

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
