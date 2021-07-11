import sys
import glob
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
from model.cls_model import ClsModel
from model.vgg3d import VGG3D

# tf.config.experimental_run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=20, help='batch size')
parser.add_argument('-mbs', '--minibatch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-ebs', '--eval_batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-rs', '--resize', type=tuple, default=None, help='target size')
parser.add_argument('-out', '--output_path', type=str, default='agecls/age_reg_96_nbn_b50_rankfake_wgan_1')
parser.add_argument('-orgs', '--fake_path', type=str, default='wgan_reg_10k_640_zm_mm_batch20_bn0_dp0_a10_b0.1_la10_lr0.0001')#_640_zm_mm_batch20_bn0_dp0_a50_b0.1_lr0.0001')
args = parser.parse_args()



import platform

if platform.system() == 'Windows':
    data_path = 'results/agecls'
    output_path = 'results/' + args.output_path
if platform.system() == 'Linux':
    data_path = '/data/psxrl3/results/agecls'
    output_path = '/data/psxrl3/results/' + args.output_path

org_path = data_path + '/processed_data'
fake_path = data_path + '/' + args.fake_path
# load filenames

real_train_list = glob.glob(org_path + '/train/*.npy')
fake_train_list = glob.glob(fake_path + '/fake_rand/*.npy')

train_list = (real_train_list + fake_train_list)[:940]
valid_list = glob.glob(org_path + '/valid/*.npy')
test_list = glob.glob(org_path + '/test/*.npy')

# real_train_list = glob.glob(fake_path + '/fake_real/*.npy')
# fake_train_list = glob.glob(fake_path + '/fake_rand/*.npy')

# train_list = (real_train_list + fake_train_list)[:950]
# valid_list = glob.glob(fake_path + '/fake_real_valid/*.npy')
# test_list = glob.glob(fake_path + '/fake_real_test/*.npy')

org_suffix = '.npy'
lab_suffix = '_lab.txt'

# org_suffix = '.nii.gz'
# lab_suffix = '_lab.txt'

pre = {org_suffix: [('channelcheck', 1)]}
# processor = Processor()
processor = SimpleImageProcessor(pre=pre)

train_provider = DataProvider(train_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        # temp_dir=output_path,
                        is_shuffle=True,
                        processor=processor)

valid_provider = DataProvider(valid_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        # temp_dir=output_path,
                        processor=processor)

# build model
vgg3d = VGG3D(n_layer=5, root_filters=16, use_bn=False)
model = ClsModel(vgg3d, org_suffix, lab_suffix, dropout=0.1)
lr = StepDecayLearningRate(learning_rate=args.learning_rate, 
                           decay_step=10,
                           decay_rate=0.8,
                           data_size=train_provider.size,
                           batch_size=args.batch_size)
optimizer = tf.keras.optimizers.Adam(lr)
trainer = Trainer(model)

# train
results = trainer.train(train_provider, valid_provider,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       mini_batch_size=args.minibatch_size,
                       output_path=output_path,
                       optimizer=optimizer,
                       learning_rate=lr,
                       eval_frequency=args.eval_batch_size)

# eval
test_provider = DataProvider(test_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        processor=processor)
trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size, need_preds=True, need_all_info=True)

with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/final_results.mat', eval_dcit)

# eval on train
test_provider = DataProvider(train_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        processor=processor)
trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size, need_preds=True, need_all_info=True)

with open(output_path + '/test_eval_ontrain.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/final_results_ontrain.mat', eval_dcit)

# eval on valid
valid_provider = DataProvider(valid_list, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        processor=processor)
trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval(valid_provider, batch_size=args.eval_batch_size, need_preds=True, need_all_info=True)

with open(output_path + '/valid_eval.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/valid_final_results.mat', eval_dcit)
print()


