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


import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from functions_preprocessing import english_standardization, visualize_prediction
from models import build_model
import soundfile


def get_all_type_paths(file_dir, _type):
    _file_paths = []
    for root_dir, sub_dir, files in os.walk(file_dir):
        for _file in files:
            if _file.endswith(_type):
                _file_paths.append(os.path.join(root_dir, _file))
    return _file_paths


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datafolder", default='../data_CTC_pre_2',
                    type=str, help="Path to data folder.")
parser.add_argument("-r", "--resultfolder", default='../result1ctc1',
                    type=str, help="Path to result folder.")
parser.add_argument("-n", "--num_samples", default=10,
                    type=int, help="Number of samples to compute prediction.")
parser.add_argument("-o", "--offset", default=0, type=int,
                    help="Offset number of samples to skip in the dataset.")
parser.add_argument("--noise_aug", default=0.0, type=float,
                    help="Noise augmentation rate (from 0 to 1).")
parser.add_argument("--audiolen", default=122792, type=int, help="audio len")
parser.add_argument("-p", "--performpath", default='',
                    type=str, help="Path to perform folder.")
args = parser.parse_args()

# --------------------------------------------
#  L O A D   P A R A M E T E R S
# --------------------------------------------
with open(os.path.join(args.datafolder, "parameters.pickle"), 'rb') as f:
    dataset_params = pickle.load(f)
with open(os.path.join(args.resultfolder, "parameters.pickle"), 'rb') as f:
    train_params = pickle.load(f)
with open(os.path.join(args.datafolder, "noise_aug.pickle"), 'rb') as f:
    noise_settings = pickle.load(f)

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

text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=english_standardization,
    max_tokens=None,
    vocabulary=keywords)


def remove_serial_duplicates(inp_seq):
    out_seq = []  # list of lists
    for i1 in range(inp_seq.shape[0]):
        q = []
        q.append(inp_seq[i1][0].numpy())  # append first element
        for j in range(1, inp_seq.shape[1]):
            if inp_seq[i1][j].numpy() != q[-1]:
                q.append(inp_seq[i1][j].numpy())
        out_seq.append(q)
    return out_seq


def remove_specific_token(inp_seqs, tok):
    out_seqs = []
    for i2 in range(len(inp_seqs)):
        out_seqs.append([s for s in inp_seqs[i2] if s != tok])
    return out_seqs


def addblank():
    in_paths = []
    ta_texts = []
    audiolen = args.audiolen
    writepath = '../blank/'
    if not os.path.exists(writepath) :
        os.makedirs(writepath)
    if args.performpath == '' :
        with open(os.path.join(args.datafolder, "speech_commands_dict.pickle"), 'rb') as f1:
            data1 = pickle.load(f1)
        with open(os.path.join(args.datafolder, "speech_commands_edit_dict.pickle"), 'rb') as f2:
            data2 = pickle.load(f2)
        with open(os.path.join(args.datafolder, "librispeech_dict.pickle"), 'rb') as f3:
            data3 = pickle.load(f3)
        pathtemp = data1['input_path'] + data2['input_path'] + data3['input_path']
        target = data1['target_text'] + data2['target_text'] + data3['target_text']
        num_samples = len(pathtemp)
        print(num_samples)
        np.random.seed(rng_seed)
        p = np.random.permutation(num_samples)
        print("**********", p)
        input_in = [pathtemp[i] for i in p]
        target_in = [target[i] for i in p]
        num_train = int(len(input_in)*train_test_split)
        inputs = input_in[num_train:]
        ta_texts = target_in[num_train:]
        for iput in inputs:
            iput = ('../'+iput).replace("\\", "/")
            audio_bin = tf.io.read_file(iput)
            audio, sr = tf.audio.decode_wav(audio_bin, 1)
            if audiolen-len(audio) == 0:
                y2 = tf.squeeze(audio, axis=-1)
            elif audiolen-len(audio) < 0:
                y2 = tf.squeeze(audio, axis=-1)
                y2 = y2[0:audiolen]
            else:
                noise = tf.constant(
                    [[0.0], ]*(audiolen-len(audio)), dtype=np.float32)
                y1 = tf.concat([audio, noise], 0)
                y2 = tf.squeeze(y1, axis=-1)
            soundfile.write(os.path.join(
                writepath, os.path.basename(iput)), y2.numpy(), sr)
            in_paths.append(os.path.join(writepath, os.path.basename(iput)))
    else :
        inputs = get_all_type_paths(args.performpath, ".wav")
        if len(inputs) == 0:
            print('There is no wav audio in {}!'.format(args.performpath))
            print('Please change the audio in wav format!')
            exit()
        for iput in inputs:
            x1 = os.path.split(iput)[-1] 
            x2 = x1.split('.')[0] 
            x3 = x2.split('-')[1:]
            if x3 == []:
                ta_texts.append(x2)
            else:
                ta_texts.append(' '.join(x3))   
            audio_bin = tf.io.read_file(iput)
            audio, sr = tf.audio.decode_wav(audio_bin, 1)
            if audiolen-len(audio) == 0:
                y2 = tf.squeeze(audio, axis=-1)
            elif audiolen-len(audio) < 0:
                y2 = tf.squeeze(audio, axis=-1)
                y2 = y2[0:audiolen]
            else:
                noise = tf.constant(
                    [[0.0], ]*(audiolen-len(audio)), dtype=np.float32)
                y1 = tf.concat([audio, noise], 0)
                y2 = tf.squeeze(y1, axis=-1)
            soundfile.write(os.path.join(
                writepath, os.path.basename(iput)), y2.numpy(), sr)
            in_paths.append(os.path.join(writepath, os.path.basename(iput)))
    return in_paths, ta_texts


input_paths, target_texts = addblank()

ds = visualize_prediction(input_paths, target_texts, noise_settings=noise_settings,
                          noise_prob=noise_aug, feature_type=feature_type, vectorizer=text_processor,
                          sr=dataset_params['sampling_rate'], frame_len=frame_length, frame_hop=frame_step,
                          fft_len=fft_length, num_mel_bins=n_mel_bins, lower_freq=lower_freq, upper_freq=upper_freq,
                          num_mfcc=n_mfcc_bins)

# (D) load model
_, model_pred = build_model(model_num, feat_dim, num_kwd)
model_pred.load_weights(os.path.join(args.resultfolder, 'model_weights.h5'))

AC_COUNT = 0
for i, (audio_batch, feats_batch, text_batch) in enumerate(ds):
    # (a) prediction: probabilty sequence of each token
    token_prob = model_pred(feats_batch)
    tokens_pred = tf.argmax(token_prob, axis=-1)
    # (c) posterior handling 1: remove duplicates
    tokens_post = remove_serial_duplicates(tokens_pred)
    # (d) posterior handling 2: remove null token (of CTC)
    tokens_post_p1 = remove_specific_token(tokens_post, num_kwd+1)
    tokens_true_p1 = remove_specific_token(text_batch.numpy(), num_kwd+1)
    if args.performpath == '' :         
        print("tokens_post_p1", tokens_post_p1)
        print("tokens_true_p1", tokens_true_p1)
        print(str(i)+"-------------------------------")
        if ((1 in tokens_post_p1[0] and 1 in tokens_true_p1[0]) 
            or (1 not in tokens_post_p1[0] and 1 not in tokens_true_p1[0])):
            AC_COUNT = AC_COUNT+1
    else:
        if 1 in tokens_post_p1[0]:
            TIPS = 'Including keyword promotion "shengteng"'
        else:
            TIPS = 'NOT Including keyword promotion "shengteng"'
        print("{}: predict {}, {}".format(input_paths[i], tokens_post_p1, TIPS))
if args.performpath == '' :
    print("tf model accuracy:", AC_COUNT/len(ds))
