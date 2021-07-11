import os
import argparse
import numpy as np 
import tensorflow as tf 
import scipy.io as sio

from core.data_provider import DataProvider
from core.trainer_tf import Trainer
from core.data_processor import SimpleImageProcessor
from core.learning_rate import StepDecayLearningRate
from models.model_wgan_gp_reg import GANModel
from models.gan_reg_10k import Gen_encoder

# tf.config.experimental_run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=10, help='batch size')
parser.add_argument('-mbs', '--minibatch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-ebs', '--eval_batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-ef', '--eval_frequency', type=int, default=1, help='frequency of evaluation within training')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-rs', '--resize', type=tuple, default=None, help='target size')
parser.add_argument('-out', '--output_path', type=str, default='agecls/wgan_reg_10k_jac_640_zm_mm_batch20_bn0_dp0_a10_b0.1_la10_lr0.0001')
parser.add_argument('-orgs', '--saved_filelists', type=str, default='std_sex_640_160_62.mat')
args = parser.parse_args()

import platform

if platform.system() == 'Windows':
    data_path = '../data/BrainT1age/all'
    output_path = 'results/' + args.output_path
if platform.system() == 'Linux':
    data_path = '/data/psxrl3/data/BrainT1age/all'
    output_path = '/data/psxrl3/results/' + args.output_path

org_path = data_path + '/std96'
saved_filelists = data_path + '/' + args.saved_filelists
# load filenames
file_mat = sio.loadmat(saved_filelists)
train_list = file_mat['train_set']
valid_list = file_mat['validation_set']
test_list = file_mat['test_set']
# add path to filename
train_list = [org_path + '/' + x.strip() for x in train_list]
valid_list = [org_path + '/' + x for x in valid_list]
test_list = [org_path + '/' + x for x in test_list]

org_suffix = '.nii.gz'
lab_suffix = '_lab.txt'

pre = {org_suffix: [('zero-mean'), ('min-max'), ('channelcheck', 1)]}
# processor = Processor()
processor = SimpleImageProcessor(pre=pre)

train_provider = DataProvider(train_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        is_shuffle=True,
                        # temp_dir=output_path,
                        processor=processor)

valid_provider = DataProvider(valid_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        # temp_dir=output_path,
                        processor=processor)

# build model
gen = Gen_encoder(use_bn=False)

# init
init_img = train_provider(1)[org_suffix]
gen(init_img)

# restore
gen.load_weights(output_path + '/ckpt/weights.h5', by_name=True)


age_img = [[] for i in range(30)]
zy = [[] for i in range(30)]
for i in range(train_provider.size):
    data = train_provider(1)
    lab = data[lab_suffix]
    age = lab[0, 2]
    idx = int(np.round(age))
    age_img[idx].append(data[org_suffix])
    zy[idx].append(age)

if not os.path.exists(output_path + '/latent'):
    os.mkdir(output_path + '/latent')

z = [[] for i in range(30)]
for i in range(len(age_img)):
    if len(age_img[i]) <= 0:
        continue
    for img in age_img[i]:
        z[i].append(gen(img))
print()

# n = []
# for i in range(len(z)):
#     l = 0 if not z[i].shape else len(z[i])
#     n.append('{}: {}'.format(i, len(z[i])))


np.save(output_path + '/latent/z.npy', z)
np.save(output_path + '/latent/zimg.npy', age_img)
np.save(output_path + '/latent/zy.npy', zy)
# z = [np.array(tf.reduce_mean(x, 0)) for x in z]

# # z to (x,y)
# x = []
# y = []
# for i in range(len(z)):
#     if np.shape(z[i]):
#         x.append(z[i])
#         y.append(i)

# x = np.array(x)
# x = x.reshape(x.shape[0], -1)
# y = np.array(y)

# # PCA
# # from sklearn.decomposition import PCA
# # pca = PCA(n_components=None)
# # pca.fit(x)
# # x = pca.transform(x)

# # RANSAC
# # from sklearn.linear_model import RANSACRegressor
# # reg = RANSACRegressor(min_samples=15).fit(x, y)
# # print(reg.score(x,y))

# # KNN
# from sklearn.neighbors import KNeighborsClassifier
# reg = KNeighborsClassifier(n_neighbors=5).fit(x, y)



# # eval
# test_provider = DataProvider(test_list, [org_suffix, lab_suffix],
#                         is_pre_load=False,
#                         processor=processor)

# train_results = []
# train_mae = []
# for i in range(train_provider.size): 
#     data = train_provider(1)
#     img = data[org_suffix]
#     lab = data[lab_suffix]
#     age = lab[0, 2]
#     # x = pca.transform(np.array(gen(img)).reshape(1, -1))
#     x = np.array(gen(img)).reshape(1, -1)
#     preds = reg.kneighbors(x, return_distance=True)
#     pred = np.average(y[preds[1]], weights=1/preds[0])
#     train_mae.append(np.abs(pred - age))  
#     train_results.append([age, pred])

# test_results = []
# test_mae = []
# for i in range(test_provider.size): 
#     data = test_provider(1)
#     img = data[org_suffix]
#     lab = data[lab_suffix]
#     age = lab[0, 2]
#     # x = pca.transform(np.array(gen(img)).reshape(1, -1))
#     x = np.array(gen(img)).reshape(1, -1)
#     # pred = reg.predict(x)
#     preds = reg.kneighbors(x, return_distance=True)
#     pred = np.average(y[preds[1]], weights=1/preds[0])
#     # pred = y[preds[1][0][0]]
#     test_mae.append(np.abs(pred - age))  
#     test_results.append([age, pred])

# sio.savemat(output_path + '/ransac_reg.mat', {'train': train_results, 'test': test_results})
# print('train mae: {}, test mae: {}'.format(np.mean(train_mae), np.mean(test_mae)))

# print()

