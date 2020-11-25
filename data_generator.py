import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
#from torchvision import datasets, transforms
import pickle
import pandas as pd
import gzip
import math
from Visual_Model_API import config

print('config.bag_tensor_path:', config.bag_tensor_path)

class ets_instance_bags(data_utils.Dataset):
  # here a bag is a video, and the instances are the segments of this video

  def __init__(self,
               id_list_path=config.id_list_path,
               video_labels_binary_path=config.video_labels_binary_path,
               overlap_percentage=.2,
               label_class=0,
               # bag_size=446,  # no need to used, internally generated
               bag_tensor_path=config.bag_tensor_path,
               # bag_label_path='F:/ETS dataset/',
               running_mode='train',
               num_frames=3570,
               num_features=174 + 81,  # shennong+librosa #fixed
               # instance_size_seconds=1/3, #fixed, cannot set externally. model-hyperparam
               norm_per_bag=False,
               vid_len_seconds=119):

    with open(video_labels_binary_path, 'rb') as f:
      self.video_labels_binary = pickle.load(f)

    with open(id_list_path, 'rb') as f:
      id_list = pickle.load(f)

    self.running_mode = running_mode
    if self.running_mode == 'train':
      print('train_list_size:', len(id_list['data_folds']['train']))
      self.example_ids = id_list['data_folds']['train']
    elif self.running_mode == 'dev':
      print('dev_list_size:', len(id_list['data_folds']['dev']))
      self.example_ids = id_list['data_folds']['dev']
    elif self.running_mode == 'test':
      print('test_list_size:', len(id_list['data_folds']['test']))
      self.example_ids = id_list['data_folds']['test']
    elif self.running_mode == 'inference':
      print('Model-mode: Inference')
      self.example_ids = []
    else:
      print('Not train, dev or test.')
      raise Exception

    self.norm_per_bag = norm_per_bag

    self.num_frames = num_frames
    self.overlap_percentage = overlap_percentage  # ( bag_size - num_frames / instance_size ) / ( bag_size - 1 )
    self.running_mode = running_mode
    self.label_class = label_class  # 0 -> holistic rating
    self.data_size = len(self.example_ids)
    # self.bag_size = bag_size  # size of Bag
    self.instance_size_seconds = 1 / 3  # instance_size_seconds
    self.fps = self.num_frames / vid_len_seconds  # fixed?
    self.vid_len_seconds = vid_len_seconds
    self.num_features = num_features
    self.bag_tensor_path = bag_tensor_path

    self.eps = np.finfo(float).eps

    # ================= Tensor Loading ======================
    print(self.bag_tensor_path+'global_mean.npy', self.bag_tensor_path)
    self.global_mean = np.load(self.bag_tensor_path+'global_mean.npy')
    self.global_std = np.load(self.bag_tensor_path+'global_std.npy')

    print('instance_size_seconds:', self.instance_size_seconds,
          # '\ntotal number of instances:', self.bag_size,
          '\nglobal_mean.shape:', self.global_mean.shape,
          '\nglobal_std.shape:', self.global_std.shape
          )
    print(running_mode + '-bag-class created.\n\n')

  def load_tensor_from_gzip(self, path, tensor_type=''):
    # print('path:', path)
    f = gzip.open(path, 'rb')
    if tensor_type == 'numpy':
      return np.load(f)
    elif tensor_type == 'torch':
      return torch.load(f)
    else:
      print('Please set parameter "tensor_type": tensor_type=\'numpy\' or tensor_type=\'torch\'')

  def generate_label(self, index):
    if self.running_mode == 'inference':
      print('Cannot generate Label in Inference mode.')
      return
    l = self.example_ids[index]
    #         print(l, self.video_labels_binary['labels'][l]['binary_labels'][self.label_class])

    label = int(self.video_labels_binary['labels'][l]['binary_labels'][self.label_class])
    return np.array([label]).astype(np.float32)

  def generate_bag(self, obj):
    if self.running_mode == 'inference':
      features = obj
      print('Loaded feature-tensor\'s shape:', features.shape)
    else:
      index = obj
      print('generate_bag() is only callable for inference for this model.')
      return

    #         print('features.shape:', features.shape)
    if features.shape[0] != self.num_frames:
      print('less than {} frames. feature_tensor shape is: {}'.format(self.num_frames, features.shape))
      raise Exception

    # 33333333333333333333333333333333
    i = 0
    cnt = 0

    segment_positions = []
    bag = []

    #         instance_size = self.instance_size/self.fps # fixed**
    instance_size_in_frames = math.ceil(self.instance_size_seconds * self.fps)  # fixed?

    while i < self.vid_len_seconds and (
        i + self.instance_size_seconds - self.vid_len_seconds) < self.instance_size_seconds * .1:
      #             print( i, i + self.instance_size_seconds, ' | ', round(i * self.fps), round((i + self.instance_size_seconds) * self.fps) )
      segment_positions.append([i, i + self.instance_size_seconds])
      temp = np.zeros([instance_size_in_frames, features.shape[-1]])
      temp2 = features[round(i * self.fps): round(i * self.fps) + instance_size_in_frames]
      temp[: temp2.shape[0], :] = temp2
      #             print('temp.shape:', temp.shape)
      bag.append(temp)
      i += self.instance_size_seconds * .8
      cnt += 1

    self.bag_size = cnt
    # if cnt != 446:
    #   print('instance has', cnt, 'instances, expected 446.')
    # 333333333333333333

    bag = np.array(bag).astype(np.float32)
    if self.norm_per_bag:
      mean = bag.mean(axis=(0))
      std = bag.std(axis=(0))
      bag = (bag - mean) / (std + self.eps)
    else:
      bag = (bag - self.global_mean) / (self.global_std + self.eps)
    print('Bag generated. Bag_shape:', bag.shape)
    return bag, segment_positions

  def instance_length_in_seconds_from_bag_size(self, total_instances):
    t = self.vid_len_seconds / (0.8 * total_instances + 0.2)  ## 20% overlap between two consecutive instances
    return t

  def __len__(self):
    return self.data_size

  def __getitem__(self, index):
    if self.running_mode == 'inference':
      print('Cannot use dataloader in Inference mode.')
      return
    if torch.is_tensor(index):
      index = index.tolist()

    bag = self.generate_bag(index % self.data_size)

    # label = np.array(self.generate_label(index % self.data_size)).astype(np.float32)
    label = self.generate_label(index % self.data_size)

    # print('(bag.shape, label.shape):', bag.shape, label.shape)

    #         return bag.cuda(), label.cuda()
    return bag, label


