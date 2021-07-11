import os
import sys
import shutil
import argparse
import numpy as np 
import tensorflow as tf 
import scipy.io as sio

sys.path.append('git/_framework')
from core.data_provider import DataProvider
from core.trainer_tf import Trainer
from core.data_processor import SimpleImageProcessor
from core.learning_rate import StepDecayLearningRate
from model.gan_reg_10k import Gen_decoder

# tf.config.experimental_run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=10, help='batch size')
parser.add_argument('-mbs', '--minibatch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-ebs', '--eval_batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-ef', '--eval_frequency', type=int, default=1, help='frequency of evaluation within training')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-rs', '--resize', type=tuple, default=None, help='target size')
parser.add_argument('-out', '--output_path', type=str, default='agecls/wgan_10k_640_zm_mm_batch20_bn0_dp0_a10_la10_lr0.0001')
parser.add_argument('-orgs', '--saved_filelists', type=str, default='std_sex_640_160_62.mat')
args = parser.parse_args()

import platform

if platform.system() == 'Windows':
    data_path = '../data/BrainT1age/all'
    output_path = 'results/' + args.output_path
if platform.system() == 'Linux':
    data_path = '/data/psxrl3/data/BrainT1age/all'
    output_path = '/data/psxrl3/results/' + args.output_path

# load data
z = np.load(output_path + '/latent/z.npy', allow_pickle=True)
zy = np.load(output_path + '/latent/zy.npy', allow_pickle=True)
# build model
gen = Gen_decoder(1, use_bn=False)

# init
gen(z[10][0])

# restore
gen.load_weights(output_path + '/ckpt/weights.h5', by_name=True)

target_folder = 'fake_rand'
if not os.path.exists(output_path + '/' + target_folder):
    os.mkdir(output_path + '/' + target_folder)

# fake real images
# for i in range(len(z)):
#     for j in range(len(z[i])):
#         fake_age = zy[i][j]
#         sub_img = gen(z[i][j])
#         sub_lab = [0, 0, fake_age]
#         np.save('{}/{}/{}_{}.npy'.format(output_path, target_folder, i, j), sub_img[0,...,0])
#         np.savetxt('{}/{}/{}_{}_lab.txt'.format(output_path, target_folder, i, j), sub_lab)

# age mean 19
# mean_z = [np.array(tf.reduce_mean(x, 0)) for x in z]
# max_data = 50
# for i in range(len(z)):
#     if len(z[i]) > 0 and len(z[i]) < max_data:
#         last_n = i-1 if i-1 >= 0 and len(z[i-1]) > 0 else i
#         next_n = i+1 if i+1 < len(z) and len(z[i+1]) > 0 else i
#         p = mean_z[next_n] - mean_z[last_n]
#         if last_n < i < next_n:
#             z_start = mean_z[last_n] + p / 4
#             z_end = mean_z[next_n] - p / 4
#         elif last_n == i and next_n > i:
#             z_start = mean_z[last_n] - p / 2
#             z_end = mean_z[next_n] - p / 2
#         elif last_n < i and next_n == i:
#             z_start = mean_z[last_n] + p / 2
#             z_end = mean_z[next_n] + p / 2
#         else:
#             assert False
#         need_data = max_data - len(z[i])
#         for j in range(need_data):
#             sub_z = z_start + (z_end - z_start) / need_data * j 
#             fake_age = i - 0.5 + j / need_data
#             sub_img = gen(sub_z)
#             sub_lab = [0, 0, fake_age]
#             np.save('{}/fake/{}_{}.npy'.format(output_path, i, j), sub_img[0,...,0])
#             np.savetxt('{}/fake/{}_{}_lab.txt'.format(output_path, i, j), sub_lab)


# age mean 19
count_z = np.zeros(30)

max_data = 50
age_range = 1

for i in range(len(z)):
    exist_pairs = []
    rand_queue = np.array(range(len(z[i])))
    for j in range(max_data-len(z[i])):
        np.random.shuffle(rand_queue)
        flag = False
        for qi in range(len(z[i])):
            if flag:
                break
            for qj in range(1, len(z[i])):
                if flag:
                    break
                if qi != qj and (rand_queue[qi], rand_queue[qj]) not in exist_pairs and (rand_queue[qj], rand_queue[qi]) not in exist_pairs:
                    w = np.random.rand()
                    fake_z = z[i][rand_queue[qi]] * w + z[i][rand_queue[qj]] * (1-w)
                    fake_age = zy[i][rand_queue[qi]] * w + zy[i][rand_queue[qj]] * (1-w)
                    sub_img = gen(fake_z)
                    sub_lab = [0, 0, fake_age]
                    np.save('{}/fake_rand/{}_{}.npy'.format(output_path, i, j), sub_img[0,...,0])
                    np.savetxt('{}/fake_rand/{}_{}_lab.txt'.format(output_path, i, j), sub_lab)
                    count_z[i] += 1
                    exist_pairs.append((rand_queue[qi], rand_queue[qj]))
                    flag = True
                    
            

print()
        

np.savetxt('{}/count.txt'.format(output_path), count_z)

# # eval
# test_provider = DataProvider(test_list, [org_suffix, lab_suffix],
#                         is_pre_load=False,
#                         processor=processor)
# trainer.restore(output_path + '/ckpt/final')
# eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size)

print()

