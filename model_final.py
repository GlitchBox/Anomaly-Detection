import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class GatedAttention(nn.Module):
    def __init__(self, embeddingDimension=120):
        # embeddingDimension = final embedding dimesion
        super(GatedAttention, self).__init__()
        # self.L = 500
        self.L = 256 ?
        self.D = 64 ?
        self.embeddingDimension = embeddingDimension #this could be a possible hyperparameter
        self.K = 1 # final output dimension
        self.poolingPolicy = ["attention", "avg", "max"]


        """ 
        i3d layers:
         shape : NUMBER_OF_INSTANCES x 7 x 7 x 1024 -> permuted to "NUMBER_OF_INSTANCES x (1024 x 7 x 7)" 
         call sequence: 1 or 2 -> 3 -> reshape -> 4
        """
        self.i3d_opticalflow_extractor1 = nn.Sequential(
            #when pooling layer is used
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1), #??#->NUMBER_OF_INSTANCES x  512 x 5 x 5
            nn.ReLU(), # LeakyReLU try kora jete pare, khub ekta labh hobe na hoyto
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1), #->NUMBER_OF_INSTANCES x  256 x 3 x 3
            nn.ReLU()
        )
        self.i3d_opticalflow_extractor2 = nn.Sequential( # Eita default code e dewa
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1), #->NUMBER_OF_INSTANCES x  512 x 5 x 5
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1), #->NUMBER_OF_INSTANCES x  256 x 3 x 3
            nn.ReLU(),
        )
        self.i3d_opticalflow_extractor3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.i3d_opticalflow_extractor4 = nn.Sequential(
            nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension), #->NUMBER_OF_INSTANCES x  self.embeddingDimension
            nn.ReLU()
        )

        self.i3d_rgb_extractor1 = nn.Sequential(

            #When pooling layer is used
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3,stride=1), #??# needs sanity checking of the shape
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
            nn.Linear(in_features=64*3*3, out_features=self.embeddingDimension), #->NUMBER_OF_INSTANCES x  self.embeddingDimension
            nn.ReLU()
        )
        

        """ 
        openpose frame_encoder layers:
         
        """
        self.frame_dense1 = nn.Linear(in_features=25*3, out_features=self.embeddingDimension)
        self.attention_V_frame = nn.Sequential(
            # transformation before applying attention ->  Input_dim: NUMBER_OF_HUMAN x self.embeddingDimension
            nn.Linear(self.embeddingDimension, self.D), # -> NUMBER_OF_HUMAN x self.D
            # nn.Linear(90, 60),
            nn.Tanh()
        )
        self.attention_U_frame = nn.Sequential( # gating mechanism
            nn.Linear(self.embeddingDimension, self.D),
            # nn.Linear(90, 60),
            nn.Sigmoid()
        )
        self.attention_weights_frame = nn.Linear(self.D, self.K) # attention score generator


        """ openpose instance layers"""
        self.instance_dense1 = nn.Linear(in_features=120, out_features=256)
        
        self.lstm_input_dim = 256
        self.lstm_output_dim = 512
        self.lstm_layers_num = 1
        self.apply_bidirectional_lstm = False
        #should we opt for bidirectional lstm layer?
        self.instance_lstm_layer = nn.LSTM(
                                    input_size=self.lstm_input_dim, 
                                    hidden_size=self.lstm_output_dim, 
                                    num_layers=self.lstm_layers_num, 
                                    batch_first=True,
                                    bidirectional=self.apply_bidirectional_lstm
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
        self.attention_weights_instance = nn.Linear(256, self.K) # attention score generator

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

    def frame_encoder(self, single_frame, pooling='attention'): #DONE
        """
        # I'm assuming that a tensor of following shape will be passed to this method: (human_count, 25, 3)
        # human_count == Variable
        """

        #single_frame will be of shape (human_count, 25, 3)
        human_count = single_frame.shape[0]

        #single_frame will be of the size (human_count, 25, 3), here human_count is the batch_size

        H = single_frame.reshape(human_count, 25*3) # -> human_count x 75
        #ekhane "human_count" variable.
        #so amader eikhane attention aggretion korte hobe

        H = self.frame_dense1(H) #output of this will be shape (human_count, 120)

        A = None
        if pooling ==  'attention' or pooling == 'max':
            A_V = self.attention_V_frame(H) #output-> (human_count, 64)
            A_U = self.attention_U_frame(H) #output-> (human_count, 64)
            A = self.attention_weights_frame(A_V*A_U) #Gating mechanism apply korlam. output-> (human_count, 1).
            A = torch.transpose(A, 1, 0) #(1, human_count)

            if pooling == 'attention':
                A = F.softmax(A, dim=1) # softmax over human_count, (1, human_count)
                # softmax doesn't have learnable parameters, hence it need not be declared in __init__
            else:
                A_ = torch.zeros([1, human_count]).to(device)
                A_[0][A.argmax()] = 1
                A = A_
        elif pooling == 'avg':
            A = torch.ones([1, human_count]) / human_count

        M = torch.mm(A, H) #(1,120) # attention apply kore fellam. output hocche M.
        return M.squeeze(0) #output shape (120)

    def openpose_instance_encoder(self, bag, instance_count, pooling="attention"): #DONE
        """
        # I'm assuming batch size will be 1 and batch_size will not be included in the input dimension
        # I'm assuming that the "single_instance" will be "list" of tensors, each with the shape -> (human_count, 25, 3)
        # len(single_instance) == the number of frame == Constant
        """
        frame_count = self.Constant_Instance_Length
        encoded_frames = torch.zeros([instance_count, frame_count, self.lstm_input_dim]) # (instance_count, frame_count, 256)

        for instance_index in range(instance_count):
            for frame_index in range(frame_count):
                encoded_frame = self.frame_encoder(bag[instance_index][frame_index]) # output shape (120)
                encoded_frame = self.instance_dense1(encoded_frame) # output shape (256)
                encoded_frames[instance_index, frame_index] = encoded_frame
        #now encoded_frames will be a tensor of shape (instance_count, frame_count, 256), because lstm expects 3d inputs

        #not passing initial activation and initial cell is the same as passing a couple of 0 vectors
        if self.USE_INSTANCE_LSTM_ENCODING:
            # option 1: lstm encoding
            activations, last_activation_cell = self.instance_lstm_layer(encoded_frames) #output shape (instance_count, frame_count, 512)
        elif self.USE_INSTANCE_CONV1D_ENCODING:
            # option 2: conv encoding
            activations = self.instance_conv1D_layer(encoded_frames)  # output shape (instance_count, frame_count, 512)
        else:
            # option 3: use nothing
            activations = encoded_frames

        outPutEmbedding = None

        if pooling == 'attention' or pooling == 'max' or pooling == 'avg':
            H = activations  # shape(instance_count, frame_count, 512)
            # A = None
            if pooling == 'attention' or pooling == 'max':
                A_V = self.attention_V_instance(H)  # output-> (instance_count, frame_count, 256)
                A_U = self.attention_U_instance(H)  # output-> (instance_count, frame_count, 256)
                A = self.attention_weights_instance(A_V * A_U).squeeze(2)
                # ^ Gating mechanism apply korlam. output-> (instance_count, frame_count).
                # A = torch.transpose(A.unsqueeze(2), 0, 1) # (instance_count, frame_count)

                if pooling == 'attention':
                    A = F.softmax(A, dim=-1) # softmax over frame_count, (instance_count, frame_count)
                    # softmax doesn't have learnable parameters, hence it need not be declared in __init__
                elif pooling == 'max':
                    A_ = torch.zeros([instance_count, frame_count]).to(device)
                    A_[range(0,instance_count), A.argmax(-1)] = 1
                    A = A_
            elif pooling == 'avg':
                A = torch.ones([instance_count, frame_count]) / frame_count
            outPutEmbedding = torch.bmm(A.unsqueeze(1), H)  # (instance_count, 1, frame_count)* (instance_count, frame_count, 512) = (instance_count, 1,512)
            outPutEmbedding = outPutEmbedding.squeeze(1)  # shape(instance_count, 512)
        elif self.USE_INSTANCE_LSTM_ENCODING:
            # since I'm not using the attention pooling, I'll just select last activation as the outputEmbedding
            # this is only usable when we are using LSTM encoding where the last embedding constains and encoding of
            # the whole sequence.
            outPutEmbedding = activations[:, -1, :] #shape (instance_count, 512)
        else:
            raise Exception

        outPutEmbedding = self.instance_dense2(outPutEmbedding) # a dimension reduction. Output -> shape (instance_count, 120)
        return outPutEmbedding #shape (instance_count, 120)


    def i3d_instance_encoder(self, i3d_optical, i3d_rgb):
        optical_encoding = self.feature_extractor_opticalflow_i3d(i3d_optical) #shape (instance, 120)
        rgb_encoding = self.feature_extractor_rgb_i3d(i3d_rgb) #shape (instance, 120)
        i3d_encoding = optical_encoding + rgb_encoding #shape (instance, 120)
        return i3d_encoding

    def openpose_bag_encoder(self, openpose_bag, pooling):
        """
        # Each bag is a datapoint. Each bag has multiple instances in it.
        # Each instance has multiple frames in it.
        # openpose bag is list of list of tensors
        """
        instance_count = len(openpose_bag)
        encoded_instances = self.openpose_instance_encoder(bag=openpose_bag, instance_count=instance_count) #(instance_count, 120)

        #instanceEncoding has shape (instance_count, 120)
        encoded_instances = encoded_instances.unsqueeze(0) #(1, instance_count, 120)

        # not passing initial activation and initial cell is the same as passing a couple of 0 vectors
        if self.USE_BAG_LSTM_ENCODING:
            # option 1: lstm encoding
            activations, last_activation_cell = self.bag_lstm_layer(
                encoded_instances)  # output shape (1, instance_count, bag_lstm_dim)
        elif self.USE_BAG_CONV1D_ENCODING:
            # option 2: conv encoding
            activations = self.instance_conv1D_layer(encoded_instances)  # output shape (1, instance_count, bag_lstm_dim)
        else:
            # option 3: use nothing
            activations = encoded_instances

        outPutEmbedding = None

        if pooling == 'attention' or pooling == 'max' or pooling == 'avg':
            H = activations  # shape(1, instance_count, bag_lstm_dim)
            # A = None
            if pooling == 'attention' or pooling == 'max':
                A_V = self.attention_V_bag(H)  # output-> (1, instance_count, bag_lstm_dim)
                A_U = self.attention_U_bag(H)  # output-> (1, instance_count, bag_lstm_dim)
                A = self.attention_weights_bag(A_V * A_U).squeeze(2)
                # ^ Gating mechanism apply korlam. output-> (1, instance_count).
                # A = torch.transpose(A.unsqueeze(2), 0, 1) # (1, instance_count)

                if pooling == 'attention':
                    A = F.softmax(A, dim=-1)  # softmax over instance_count, (1, instance_count)
                    # softmax doesn't have learnable parameters, hence it need not be declared in __init__
                elif pooling == 'max':
                    A_ = torch.zeros([1, instance_count]).to(device)
                    A_[0, A.argmax(-1)] = 1
                    A = A_
            elif pooling == 'avg':
                A = torch.ones([1, instance_count]) / instance_count
            outPutEmbedding = torch.mm(A, H[0])  # (1, instance_count)* (instance_count, bag_lstm_dim) = (1, bag_lstm_dim)
            outPutEmbedding = outPutEmbedding.squeeze(0)  # shape(bag_lstm_dim)
        elif self.USE_BAG_LSTM_ENCODING:
            # since I'm not using the attention pooling, I'll just select last activation as the outputEmbedding
            # this is only usable when we are using LSTM encoding where the last embedding constains the encoding of
            # the whole sequence.
            outPutEmbedding = activations[0, -1, :]  # shape (bag_lstm_dim)
        else:
            raise Exception

        outPutEmbedding = self.bag_dense(outPutEmbedding)  # a dimension reduction. Output -> shape (bag_dense_dim)

        return outPutEmbedding  # shape (bag_dense_dim)

    def i3d_bag_encoder(self, i3d_optical, i3d_rgb, pooling):
        """
        # Each bag is a datapoint. Each bag has multiple instances in it.
        # Each instance has multiple frames in it.
        # openpose bag is list of list of tensors
        """
        instance_count = i3d_optical.shape[0]
        encoded_instances = self.i3d_instance_encoder(i3d_optical=i3d_optical,
                                                      i3d_rgb=i3d_rgb)  # (instance_count, 120)

        #instanceEncoding has shape (instance_count, 120)
        encoded_instances = encoded_instances.unsqueeze(0) #(1, instance_count, 120)

        # not passing initial activation and initial cell is the same as passing a couple of 0 vectors
        if self.USE_BAG_LSTM_ENCODING:
            # option 1: lstm encoding
            activations, last_activation_cell = self.bag_lstm_layer(
                encoded_instances)  # output shape (1, instance_count, bag_lstm_dim)
        elif self.USE_BAG_CONV1D_ENCODING:
            # option 2: conv encoding
            activations = self.instance_conv1D_layer(encoded_instances)  # output shape (1, instance_count, bag_lstm_dim)
        else:
            # option 3: use nothing
            activations = encoded_instances

        outPutEmbedding = None

        if pooling == 'attention' or pooling == 'max' or pooling == 'avg':
            H = activations  # shape(1, instance_count, bag_lstm_dim)
            # A = None
            if pooling == 'attention' or pooling == 'max':
                A_V = self.attention_V_bag(H)  # output-> (1, instance_count, bag_lstm_dim)
                A_U = self.attention_U_bag(H)  # output-> (1, instance_count, bag_lstm_dim)
                A = self.attention_weights_bag(A_V * A_U).squeeze(2)
                # ^ Gating mechanism apply korlam. output-> (1, instance_count).
                # A = torch.transpose(A.unsqueeze(2), 0, 1) # (1, instance_count)

                if pooling == 'attention':
                    A = F.softmax(A, dim=-1)  # softmax over instance_count, (1, instance_count)
                    # softmax doesn't have learnable parameters, hence it need not be declared in __init__
                elif pooling == 'max':
                    A_ = torch.zeros([1, instance_count]).to(device)
                    A_[0, A.argmax(-1)] = 1
                    A = A_
            elif pooling == 'avg':
                A = torch.ones([1, instance_count]) / instance_count
            outPutEmbedding = torch.mm(A, H[0])  # (1, instance_count)* (instance_count, bag_lstm_dim) = (1, bag_lstm_dim)
            outPutEmbedding = outPutEmbedding.squeeze(0)  # shape(bag_lstm_dim)
        elif self.USE_BAG_LSTM_ENCODING:
            # since I'm not using the attention pooling, I'll just select last activation as the outputEmbedding
            # this is only usable when we are using LSTM encoding where the last embedding constains the encoding of
            # the whole sequence.
            outPutEmbedding = activations[0, -1, :]  # shape (bag_lstm_dim)
        else:
            raise Exception

        outPutEmbedding = self.bag_dense(outPutEmbedding)  # a dimension reduction. Output -> shape (bag_dense_dim)

        return outPutEmbedding  # shape (bag_dense_dim)

    def joint_openpose_and_i3d_bag_encoder(self, i3d_optical, i3d_rgb, openpose_bag, pooling):
        """
        # Each bag is a datapoint. Each bag has multiple instances in it.
        # Each instance has multiple frames in it.
        # openpose bag is list of list of tensors
        """
        instance_count = len(openpose_bag)
        openpose_encoded_instances = self.openpose_instance_encoder(bag=openpose_bag,
                                                               instance_count=instance_count)  # (instance_count, 120)
        i3d_encoded_instances = self.i3d_instance_encoder(i3d_optical=i3d_optical,
                                                          i3d_rgb=i3d_rgb)  # (instance_count, 120)

        encoded_instances = torch.cat([openpose_encoded_instances, i3d_encoded_instances])

        # instanceEncoding has shape (instance_count, 120)
        encoded_instances = encoded_instances.unsqueeze(0)  # (1, instance_count, 120)

        # not passing initial activation and initial cell is the same as passing a couple of 0 vectors
        if self.USE_BAG_LSTM_ENCODING:
            # option 1: lstm encoding
            activations, last_activation_cell = self.bag_lstm_layer(
                encoded_instances)  # output shape (1, instance_count, bag_lstm_dim)
        elif self.USE_BAG_CONV1D_ENCODING:
            # option 2: conv encoding
            activations = self.instance_conv1D_layer(
                encoded_instances)  # output shape (1, instance_count, bag_lstm_dim)
        else:
            # option 3: use nothing
            activations = encoded_instances

        outPutEmbedding = None

        if pooling == 'attention' or pooling == 'max' or pooling == 'avg':
            H = activations  # shape(1, instance_count, bag_lstm_dim)
            # A = None
            if pooling == 'attention' or pooling == 'max':
                A_V = self.attention_V_bag(H)  # output-> (1, instance_count, bag_lstm_dim)
                A_U = self.attention_U_bag(H)  # output-> (1, instance_count, bag_lstm_dim)
                A = self.attention_weights_bag(A_V * A_U).squeeze(2)
                # ^ Gating mechanism apply korlam. output-> (1, instance_count).
                # A = torch.transpose(A.unsqueeze(2), 0, 1) # (1, instance_count)

                if pooling == 'attention':
                    A = F.softmax(A, dim=-1)  # softmax over instance_count, (1, instance_count)
                    # softmax doesn't have learnable parameters, hence it need not be declared in __init__
                elif pooling == 'max':
                    A_ = torch.zeros([1, instance_count]).to(device)
                    A_[0, A.argmax(-1)] = 1
                    A = A_
            elif pooling == 'avg':
                A = torch.ones([1, instance_count]) / instance_count
            outPutEmbedding = torch.mm(A,
                                       H[0])  # (1, instance_count)* (instance_count, bag_lstm_dim) = (1, bag_lstm_dim)
            outPutEmbedding = outPutEmbedding.squeeze(0)  # shape(bag_lstm_dim)
        elif self.USE_BAG_LSTM_ENCODING:
            # since I'm not using the attention pooling, I'll just select last activation as the outputEmbedding
            # this is only usable when we are using LSTM encoding where the last embedding constains the encoding of
            # the whole sequence.
            outPutEmbedding = activations[0, -1, :]  # shape (bag_lstm_dim)
        else:
            raise Exception

        outPutEmbedding = self.bag_dense(outPutEmbedding)  # a dimension reduction. Output -> shape (bag_dense_dim)

        return outPutEmbedding  # shape (bag_dense_dim)

    def final_encoding_generator(self, i3d_optical, i3d_rgb, openpose_bag, pooling):
        """
        :param i3d_optical: 
        :param i3d_rgb: 
        :param openpose_bag: 
        :param pooling: 
        :return:
        
        possible experiments: 
        1. Both features:
            a. Merged before bag level MIL pooling - early merging
            b. Merged before classification - late merging
        2. Single feature:
            a. i3d
            b. openpose
        3. Ensembling: Train the Single-featured-models individually and then average their outputs
        4. Fusion modeling:
            a. Early fusion: ??
            b. Late fusion: Train the Single-featured-models individually and then jointly train them (maybe very low
            learning rate) with an extra randomly initiated classifier (normal learning rate) == kinda' transfer learning
        """
        if config.FEATURE_SET=='both' and not config.MERGE_BEFORE_CLASSIFICATION:
            return self.joint_openpose_and_i3d_bag_encoder(
                        i3d_optical=i3d_optical,
                        i3d_rgb=i3d_rgb,
                        openpose_bag=openpose_bag,
                        pooling=config.bag_poolings
                    )
        if config.FEATURE_SET_TYPE=='openpose' or config.MERGE_BEFORE_CLASSIFICATION:
            openpose_bag_encoding_vector, attention_on_instances, attention_on_frames, \
            attention_on_human_pose_encoding = self.openpose_bag_encoder(openpose_bag=openpose_bag,
                                                                 pooling=config.bag_pooling) #shape (120)
            final_bag_encoding_vector = openpose_bag_encoding_vector
        if config.FEATURE_SET=='i3d' or config.MERGE_BEFORE_CLASSIFICATION:
            i3d_bag_encoding_vector, attention_on_instances, attention_on_frames, \
            attention_on_human_pose_encoding = self.i3d_bag_encoder(i3d_optical=i3d_optical,
                                                  i3d_rgb=i3d_rgb,
                                                  pooling=config.bag_pooling)
            final_bag_encoding_vector = i3d_bag_encoding_vector
        if config.MERGE_BEFORE_CLASSIFICATION:
            final_bag_encoding_vector = torch.cat([openpose_bag_encoding_vector, i3d_bag_encoding_vector])

        return final_bag_encoding_vector, attention_on_instances, attention_on_frames, \
                attention_on_human_pose_encoding
    #I'm assuming x will be like the following [i3d_optical, i3d_rgb, openpose_list]
    #openpose_list ==  list of instances
    #an instance == list of frames
    #a frame == a tensor of shape (human_count, 25, 3)
    def forward(self, joint_bag):
        # x = x.squeeze(0)
        i3d_optical = joint_bag[0]
        i3d_rgb = joint_bag[1]
        openpose_bag = joint_bag[2]

        final_bag_encoding_vector, attention_on_instances, attention_on_frames, \
        attention_on_human_pose_encoding = self.joint_openpose_and_i3d_bag_encoder(
                                                i3d_optical=i3d_optical,
                                                i3d_rgb=i3d_rgb,
                                                openpose_bag=openpose_bag,
                                                pooling=config.bag_poolings
                                            )

        Y_prob = self.classifier(final_bag_encoding_vector)

        return Y_prob, attention_on_instances, attention_on_frames, attention_on_human_pose_encoding

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return error, Y_hat
    #
    # def calculate_objective(self, X, Y):
    #     Y = Y.float()
    #     Y_prob, _, A = self.forward(X)
    #     Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    #     neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    #
    #     return neg_log_likelihood, A
