# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import pickle
from models import build_model
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultfolder", default='../result1ctc1',
                    type=str, help="Path to result folder.")
parser.add_argument("-d", "--datafolder", default='../data_CTC_pre_2',
                    type=str, help="Path to data folder")
args = parser.parse_args()
with open(os.path.join(args.datafolder, "parameters.pickle"), 'rb') as f:
    dataset_params = pickle.load(f)
with open(os.path.join(args.resultfolder, "parameters.pickle"), 'rb') as f:
    train_params = pickle.load(f)
keywords = dataset_params['keywords']
num_kwd = dataset_params['num_kwd']
# dataset partitioning and shuffling
model_num = train_params['model_num']
feature_type = train_params['feature_type']
train_test_split = train_params['train_test_split']
rng_seed = train_params['rng_seed']
noise_aug = train_params['noise_aug']
# STFT SPECIFICATION
frame_length = train_params['frame_length']
frame_step = train_params['frame_step']
fft_length = train_params['fft_length']
# MFCC SPECIFICATION
lower_freq = train_params['lower_freq']
upper_freq = train_params['upper_freq']
n_mel_bins = train_params['n_mel_bins']
n_mfcc_bins = train_params['n_mfcc_bins']
if feature_type == 'mfcc':
    feat_dim = n_mfcc_bins
elif feature_type == 'melspec':
    feat_dim = n_mel_bins
_, model_pred = build_model(model_num, feat_dim, num_kwd)
model_pred.load_weights(os.path.join(args.resultfolder, 'model_weights.h5'))
tf.saved_model.save(model_pred, "tmp_model")
