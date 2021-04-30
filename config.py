"""
    Bools
"""
MERGE_BEFORE_CLASSIFICATION = True

"""
General variables
"""
Constant_Instance_Length = 10 
L = 256
D = 64
embeddingDimension = 120
K = 1
FEATURE_SET_TYPE=='both'
I3D_FEATURESET  = "both"

"""
i3d layers
"""
i3d_opt_ext1 = {
    "pool1_kernel": 3,
    "pool1_stride": 1,
    "conv1_in" : 1024,
    "conv1_out" : 512,
    "conv1_kernel": 3,
    "conv1_stride": 1,
    "conv2_in": 512,
    "conv2_out": 256,
    "conv2_kernel": 1,
    "conv2_stride": 1
}

i3d_opt_ext2 = {
    "conv1_in": 1204,
    "conv1_out": 512,
    "conv1_kernel": 3,
    "conv1_stride": 1,
    "conv2_in": 512,
    "conv2_out": 256,
    "conv2_kernel": 3,
    "conv2_stride": 1
}

i3d_opt_ext3 = {
    "conv1_in":256,
    "conv1_out":64,
    "conv1_kernel":1
}

i3d_opt_ext4 = {
    "dense1_in": 64*3*3,
    "dense1_out": embeddingDimension
}

i3d_rgb_ext1 = {
    "pool1_kernel": 3,
    "pool1_stride": 1,
    "conv1_in" : 1024,
    "conv1_out" : 512,
    "conv1_kernel": 3,
    "conv1_stride": 1,
    "conv2_in": 512,
    "conv2_out": 256,
    "conv2_kernel": 1,
    "conv2_stride": 1
}

i3d_rgb_ext2 = {
    "conv1_in" : 1024,
    "conv1_out" : 512,
    "conv1_kernel": 3,
    "conv1_stride": 1,
    "conv2_in": 512,
    "conv2_out": 256,
    "conv2_kernel": 1,
    "conv2_stride": 1
}

i3d_rgb_ext3 = {
    "conv1_in" : 256,
    "conv1_out" : 64,
    "conv1_kernel": 1,
    "conv1_stride": 1,
}

i3d_rgb_ext4 = {
    "dense1_in": 64*3*3,
    "dense1_out": embeddingDimension
}

"""
openpose frame_encoder layers
"""

frame_dense1 = {
    "in": 25*3,
    "out":embeddingDimension
}

frame_V = {
    "in": embeddingDimension,
    "out": D
}

frame_U = {
    "in": embeddingDimension,
    "out":D
}

frame_attention_weights = {
    "in": D,
    "out": K
}
"""
openpose_instance_encoder
"""
USE_INSTANCE_LSTM_ENCODING = True
USE_INSTANCE_CONV1D_ENCODING = False

instance_dense1 = {
    "in": 120,
    "out": 256
}

instance_lstm1 = {
    "in":256,
    "out":512,
    "layers":1,
    "ifBidirectional": False
}

instance_dense2 = {
    "in":512,
    "out": 120
}

instance_V = {
    "in": 512,
    "out":256
}

instance_U = {
    "in": 512,
    "out": 256
}

instance_attention_weight = {
    "in":256,
    "out":K
}

instance_conv1 = {
    "in": 512,
    "out":512,
    "kernel":Constant_Instance_Length//2,
    "stride": 1,
    "padding": Constant_Instance_Length//4
}

"""
openpose_bag_encoder
"""
USE_BAG_LSTM_ENCODING = True
USE_BAG_CONV1D_ENCODING = False

bag_lstm1 = {
    "in": 120,
    "out": 512,
    "layers":1,
    "ifBidirectional": False
}

bag_V = {
    "in": 512,
    "out":256
}

bag_U = {
    "in": 512,
    "out": 256
}

bag_attention_weight = {
    "in":256,
    "out":1
}

bag_dense1 = {
    "in": 512,
    "out":256
}

"""
classifier layers
"""

classifier_dense1 = {
    "in": L*K,
    "out": 1
}
