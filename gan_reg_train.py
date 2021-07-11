import os
import sys
import shutil
import argparse
import numpy as np 
import tensorflow as tf 
import scipy.io as sio

sys.path.append('git/_framework')
from utils import util as U
from core.data_provider import DataProvider
from core.trainer_tf import Trainer
from core.data_processor import SimpleImageProcessor
from core.learning_rate import StepDecayLearningRate
from models.model_wgan_gp_reg import GANModel
from models.gan_reg import Generator, Discriminator, GEN_REG
from models.vgg3d import VGG3D

# tf.config.experimental_run_functions_eagerly(True)

epochs = 100
batch_size = 20
mini_batch_size = 1
eval_batch_size = 5
eval_frequency = 1
use_bn = False
use_res = True
learning_rate = 0.0001
g_alpha = 10
g_beta = 0.1
g_lambda = 10
dropout = 0
resize = None
output_path = 'output/gan'
saved_filelists = 'std_sex_640_160_62.mat'

import platform

if platform.system() == 'Windows':
    data_path = '../data/BrainT1age/all'
    output_path = 'results/' + output_path

# if os.path.exists(output_path):
#     shutil.rmtree(output_path, ignore_errors=True)

org_path = data_path + '/std'
saved_filelists = data_path + '/' + saved_filelists
# load filenames
file_mat = sio.loadmat(saved_filelists)
train_list = file_mat['train_set']
valid_list = file_mat['validation_set']
test_list = file_mat['test_set']
# add path to filename

org_suffix = '.nii.gz'
age_suffix = '_lab.txt'

pre = {org_suffix: [('zero-mean'), ('min-max'), ('channelcheck', 1)]}
# processor = Processor()
processor = SimpleImageProcessor(pre=pre)

train_provider = DataProvider(train_list, [org_suffix, age_suffix],
                        is_pre_load=False,
                        is_shuffle=True,
                        # temp_dir=output_path,
                        processor=processor)

valid_provider = DataProvider(valid_list, [org_suffix, age_suffix],
                        is_pre_load=False,
                        # temp_dir=output_path,
                        processor=processor)

# build model
gen = Generator(output_channels=1, use_bn=use_bn)
disc = Discriminator(use_bn=use_bn)
# gen = Generator(n_class=1, n_layer=6, root_filters=16, use_bn=use_bn, use_res=use_res)
# disc = Discriminator(n_class=1, n_layer=5, root_filters=16, use_bn=use_bn, use_res=use_res)
reg = VGG3D(n_layer=5, root_filters=16, use_bn=use_bn)
gen_reg = GEN_REG(gen, reg)
model = GANModel([gen_reg, disc], org_suffix, org_suffix, age_suffix, g_alpha=g_alpha, g_beta=g_beta, g_lambda=g_lambda, dropout=dropout)
gen_lr = StepDecayLearningRate(learning_rate=learning_rate, 
                           decay_step=10,
                           decay_rate=0.8,
                           data_size=train_provider.size,
                           batch_size=batch_size)
disc_lr = StepDecayLearningRate(learning_rate=learning_rate, 
                           decay_step=10,
                           decay_rate=0.8,
                           data_size=train_provider.size,
                           batch_size=batch_size)
gen_optimizer = tf.keras.optimizers.Adam(gen_lr)
disc_optimizer = tf.keras.optimizers.Adam(disc_lr)
trainer = Trainer(model)

# train
results = trainer.train(train_provider, valid_provider,
                       epochs=epochs,
                       batch_size=batch_size,
                       mini_batch_size=mini_batch_size,
                       output_path=output_path,
                       optimizer=[gen_optimizer, disc_optimizer],
                       learning_rate=[gen_lr, disc_lr],
                       eval_frequency=eval_frequency)

# eval
test_provider = DataProvider(test_list, [org_suffix, age_suffix],
                        is_pre_load=False,
                        processor=processor)
trainer.restore(output_path + '/ckpt/final')


eval_dcit = trainer.eval(test_provider, batch_size=eval_batch_size, need_preds=True, need_all_info=True)

gen.save_weights(output_path + '/ckpt/weights.h5')

with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/final_results.mat', eval_dcit)
print()

# eval on train
test_provider = DataProvider(train_list, [org_suffix, age_suffix],
                        is_pre_load=False,
                        processor=processor)
trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval(test_provider, batch_size=eval_batch_size, need_preds=True, need_all_info=True)

with open(output_path + '/test_eval_ontrain.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/final_results_ontrain.mat', eval_dcit)
print()

