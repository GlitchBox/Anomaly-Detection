
"""
General variables
"""
Constant_Instance_Length = 10 ?
L = 256
D = 64
embeddingDimension = 120
K = 1

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

"""
openpose_instance_encoder
"""
USE_INSTANCE_LSTM_ENCODING = True
USE_INSTANCE_CONV1D_ENCODING = False

"""
openpose_bag_encoder
"""
USE_BAG_LSTM_ENCODING = True
USE_BAG_CONV1D_ENCODING = False?
