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

        """ i3d layers """
        self.i3d_opticalflow_extractor1 = nn.Sequential(

            #when pooling layer is used
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

            #When pooling layer is used
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
        

        """ openpose frame layers """
        self.frame_dense1 = nn.Linear(in_features=25*3, out_features=self.embeddingDimension)
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


        """ openpose instance layers"""
        self.instance_dense1 = nn.Linear(in_features=120, out_features=256)
        
        self.lstm_input_dim = 256
        self.lstm_output_dim = 512
        self.lstm_layers_num = 1
        #should we opt for bidirectional lstm layer?
        self.instance_lstm_layer = nn.LSTM(
                                    input_size=self.lstm_input_dim, 
                                    hidden_size=self.lstm_output_dim, 
                                    num_layers=self.lstm_layers_num, 
                                    batch_first=True,
                                    # bidirectional=True
                                    )
        self.instance_dense2 = nn.Linear(in_features=512, out_features=120)
        
        self.attention_V_instance = nn.Sequential(
            nn.Linear(512, 256),
            # nn.Linear(90, 60),
            nn.Tanh()
        )
        self.attention_U_instance = nn.Sequential(
            nn.Linear(512, 256),
            # nn.Linear(90, 60),
            nn.Sigmoid()
        )
        self.attention_weights_instance = nn.Linear(256, self.K)

        "openpose bag layers"

        """classfier layers"""
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
        # rgb = rgb.permute(0,3,1,2)

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

    #I'm assuming that a tensor of following shape will be passed to this method: (m, human_count,25, 3)
    #here m == number of frames in an instance
    #m will work as the batch size for the following calculations
    def frame_encoder(self, openpose_instance_frames, pooling='attention'):

        #openpose_instance_frames will be of shape (human_count, 25, 3)
        # human_count = openpose_instance_frames.shape[0]
        
        #openpose_instance_frames will be of the size (m, human_count, 25, 3), here m is the batch_size
        m = openpose_instance_frames.shape[0]
        # human_count = openpose_instance_frames.shape[1]

        # H = openpose_instance_frames.reshape(human_count, 25*3)
        H = openpose_instance_frames.reshape(-1, human_count, 25*3)
        H = self.frame_dense1(H) #output of this will be shape (human_count, 120)

        A = None
        if pooling ==  'attention' or pooling == 'max':
            A_V = self.attention_V_frame(H) #(human_count, 64)
            A_U = self.attention_U_frame(H) #(human_count, 64)
            A = self.attention_weights_frame(A_V*A_U) # (human_count, 1)
            A = torch.transpose(A, 1, 0) #(1, human_count)
            # A = A.permute(0, 2, 1) #(m, 1, human_count)
            
            if pooling == 'attention':
                A = F.softmax(A, dim=1) # softmax over human_count, (1, human_count)
                # A = F.softmax(A, dim=2) #softmax over human_count (m, 1, human_count), softmax doesn't have learnable parameters, hence it need not be declared in __init__
            else:
                # A_ = torch.zeros((m, 1, human_count))
                A_ = torch.zeros(( 1,human_count))
                A_[0][A.argmax()] = 1
                # for i in range(m):
                #     A_[i][0][A[i][0].argmax()] = 1
                A = A_
        elif pooling == 'avg':
            # A = torch.ones((m,1,human_count))/human_count
            A = torch.ones((1,human_count))/human_count
        
        # M = torch.zeros((m,1,120))
        M = torch.mm(A, H) #(1,120) 
        # for i in range(m):
        #     M[i] = torch.mm(A[i], H[i]) #Shape of M (m,1,120)
        return M.squeeze(0) #output shape (120)
        # return M.squeeze(1) #output shape (m,120)

    #I'm assuming batch size will be 1 and batch_size will not be included in the input dimension
    #I'm assuming that the single_instance will be "list" of tensors of shape (human_count, 25, 3)
    def openpose_instance_encoder(self, single_instance, pooling="attention"):
        
        instanceLen = len(single_instance)
        encoded_frames = torch.zeros((instanceLen, self.lstm_input_dim)) #(frame_count, 256)

        for i in range(instanceLen):
            encoded_frame = self.frame_encoder(single_instance[i]) #output shape (120)
            encoded_frame = self.instance_dense1(encoded_frame) #output shape (256)
            # encoded_frame = encoded_frame.unsqueeze(0) #output shape (1,256)
            # encoded_frame = encoded_frame.unsqueeze(1) #output shape (m,1,256)
            encoded_frames[i] = encoded_frame
        encoded_frames = encoded_frames.unsqueeze(0)
        #now encoded_frames will be a tensor of shape (1, frame_count, 256), because lstm expects 3d inputs
        
        #not passing initial activation and initial cell is the same as passing a couple of 0 vectors
        activations, last_activation_cell = self.instance_lstm_layer(encoded_frames) #output shape (1, frame_count, 512)
        outPutEmbedding = None
        
        if pooling == "attention":
            H = activations[0] #shape(frame_count, 512)
            A_V =  self.attention_V_instance(H) #shape (frame_count, 256)
            A_U = self.attention_U_instance(H) #shape (frame_count, 256)
            A = self.attention_weights_instance(A_V*A_U) #shape (frame_count, 1)
            A = torch.transpose(A, 1, 0) #shape (1, frame_count)
            A = F.softmax(A, dim=1) # softmax over frame_count, (1, frame_count)
            outPutEmbedding = torch.mm(A,H) #shape (1,512)
            outPutEmbedding = outPutEmbedding.squeeze(0) #shape(512)
        else:
            #since I'm not using the attention pooling, I'll just select last activation as the outputEmbedding
            outPutEmbedding = activations[0][-1] #shape (512)
        
        outPutEmbedding = self.instance_dense2(outPutEmbedding) #shape (120)
        return outPutEmbedding #shape (120)
        #return outPutEmbedding[0][0] #output shape (120)
        # return outPutEmbedding[0] #output shape (m,120)

    #Each bag is a datapoint. Each bag has multiple instance in it. Each instance has multiple
    #frames in it.
    #openpose bag is list of list of tensors
    def openpose_bag_encoder(self, bag, pooling="attention"):

        instanceNum = len(bag)
        instanceEncoding = torch.zeros((instanceNum, self.embeddingDimension)) #(instanceNum, 120)

        for idx in range(instanceNum):
            singleInstance = self.openpose_instance_encoder(bag[idx])
            instanceEncoding[idx] = singleInstance 
        #instanceEncoding has shape (instanceNum, 120)
        instanceEncoding = instanceEncoding.unsqueeze(0) #(1, instanceNum, 120)
        
        #not passing initial activation and initial cell is the same as passing a couple of 0 vectors
        # activations, last_activation_cell = self.instance_lstm_layer(instanceEncoding) #output shape (1, frame_count, 512)
        # outPutEmbedding = None
        
        # if pooling == "attention":
        #     H = activations[0] #shape(frame_count, 512)
        #     A_V =  self.attention_V_instance(H) #shape (frame_count, 256)
        #     A_U = self.attention_U_instance(H) #shape (frame_count, 256)
        #     A = self.attention_weights_instance(A_V*A_U) #shape (frame_count, 1)
        #     A = torch.transpose(A, 1, 0) #shape (1, frame_count)
        #     A = F.softmax(A, dim=1) # softmax over frame_count, (1, frame_count)
        #     outPutEmbedding = torch.mm(A,H) #shape (1,512)
        #     outPutEmbedding = outPutEmbedding.squeeze(0) #shape(512)
        # else:
        #     #since I'm not using the attention pooling, I'll just select last activation as the outputEmbedding
        #     outPutEmbedding = activations[0][-1] #shape (512)
        
        # outPutEmbedding = self.instance_dense2(outPutEmbedding) #shape (120)
        # return outPutEmbedding #shape (120)

    #I'm assuming x will be like the following [i3d_optical, i3d_rgb, openpose_list]
    #openpose_list ==  list of instances
    #an instance == list of frames
    #a frame == a tensor of shape (human_count, 25, 3)
    def forward(self, x, y):
        # x = x.squeeze(0)
        i3d_optical = x[0]
        i3d_rgb = x[1]
        openpose_bag = x[2]

        optical_encoding = self.feature_extractor_opticalflow_i3d(i3d_optical) #shape (120)
        rgb_encoding = self.feature_extractor_rgb_i3d(i3d_rgb) #shape (120)
        i3d_encoding = optical_encoding + rgb_encoding #shape (120)

        openpose_encoding = self.openpose_bag_encoder(openpose_bag) #shape (120)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL

        # A_V = self.attention_V(H)  # NxD
        # A_U = self.attention_U(H)  # NxD
        # A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N

        # M = torch.mm(A, H)  # KxL

        # Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

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
