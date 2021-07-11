import numpy as np
import tensorflow as tf
from abc import ABCMeta,abstractmethod

from utils import eval_methods as EM
from utils import loss_tf as LF
from utils import util as U
from utils.process_methods import one_hot
from models.model import Model


class GANModel(Model):
    def __init__(self, net, x_suffix, y_suffix, age_suffix=None, g_alpha=10, g_beta=0.1, g_lambda=10, dropout=0):
        super().__init__(net)
        self._x_suffix = x_suffix
        self._y_suffix = y_suffix
        self._age_suffix = age_suffix
        self._alpha = g_alpha
        self._beta = g_beta
        self._lambda = g_lambda
        self.dropout = dropout

    def get_grads(self, data_dict):
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]
        gen_reg = self.net[0]
        disc = self.net[1]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_logits, reg_logits = gen_reg(xs, self.dropout, True)
            disc_gen_logits = disc(gen_logits, self.dropout, True)
            disc_real_logits = disc(ys, self.dropout, True)

            gen_loss = self._get_gen_loss(disc_gen_logits, gen_logits, reg_logits, data_dict)[0]
            disc_loss = self._get_disc_loss(disc_real_logits, disc_gen_logits, gen_logits, ys, True)[0] 

        gen_grads = gen_tape.gradient(gen_loss, self.net[0].trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.net[1].trainable_variables)
        return gen_grads, disc_grads

    def eval(self, data_dict, **kwargs):
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]

        gen_reg = self.net[0]
        disc = self.net[1]
        
        gen_logits, reg_logits = gen_reg(xs, 0., False)
        disc_gen_logits = disc(gen_logits, 0., False)
        disc_real_logits = disc(ys, 0., False)

        total_gen_loss, gen_loss, l1_loss, age_loss = self._get_gen_loss(disc_gen_logits, gen_logits, reg_logits, data_dict)
        total_disc_loss, disc_loss = self._get_disc_loss(disc_real_logits, disc_gen_logits, gen_logits, ys, False) 

        eval_results = {'loss': total_gen_loss,
                        'gen_loss': gen_loss,
                        'disc_loss': disc_loss,
                        'disc_gp_loss': total_disc_loss,
                        'gen_mae': l1_loss,
                        'age_mse': age_loss}

        need_preds = kwargs.get('need_preds', False)
        if need_preds:
            eval_results.update({'pred': reg_logits})

        need_all_info = kwargs.get('need_all_info', False)
        if need_all_info:
            all_info = data_dict[self._age_suffix]
            eval_results.update({'all_info': all_info})

        need_imgs = kwargs.get('need_imgs', None)
        if need_imgs is not None:
            eval_results.update({'imgs': self._get_imgs_eval(xs, ys, gen_logits)})

        return eval_results

    def predict(self, data_dict):
        return self.net[0](data_dict[self._x_suffix])

    def _get_gen_loss(self, disc_gen_logits, gen_logits, reg_logits, data_dict):
        ys = data_dict[self._y_suffix]
        ages = data_dict[self._age_suffix][..., 2:3]

        gan_loss = -tf.reduce_mean(disc_gen_logits)
        l1_loss = tf.losses.mae(gen_logits, ys)
        age_loss = tf.losses.mse(reg_logits, ages)
        

        gan_loss = tf.reduce_mean(gan_loss, range(1, gan_loss.ndim))
        l1_loss = tf.reduce_mean(l1_loss, range(1, l1_loss.ndim))
        age_loss = tf.reduce_mean(age_loss, range(1, l1_loss.ndim))

        # cv2.imwrite('/data/psxrl3/results/gen.png', np.array(gen_logits[0,...,50,0] * 255))
        return gan_loss + self._alpha * l1_loss + self._beta * age_loss, gan_loss, l1_loss, age_loss

    def _get_disc_loss(self, disc_real_logits, disc_gen_logits, gen_logits, ys, training): 
        alpha = tf.random.uniform(shape=[ys.shape[0], 1, 1, 1, 1], minval=0., maxval=1.)
        inter_sample = gen_logits * alpha + ys * (1 - alpha)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(inter_sample)
            inter_score = self.net[1](inter_sample, self.dropout, training)
        gp_gradients = gp_tape.gradient(inter_score, inter_sample)
        gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3, 4]))
        gp_loss = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
        disc_loss = tf.reduce_mean(disc_gen_logits) - tf.reduce_mean(disc_real_logits)

        return disc_loss + gp_loss * self._lambda, disc_loss

    def gradient_penalty(self, gen_logits, ys):
        alpha = tf.random.uniform(shape=[ys.shape[0], 1, 1, 1, 1], minval=0., maxval=1.)
        differences = gen_logits - ys
        interpolates = ys+ (alpha * differences)
        gradients = tf.gradients(self.net[1](interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        return gradient_penalty

    def _get_imgs_eval(self, xs, ys, prob):
        prob = np.array(prob)
        img_dict = {}
        n_class = ys.shape[-1]
        n_slice = 10 
        for c in range(n_class):
            imgs = None
            for n in range(ys.shape[0]):
                idx = [int(ys.shape[-2] // n_slice * (s + 0.5)) for s in range(n_slice)]
                img = U.combine_2d_imgs_from_tensor([xs[n, ..., idx, c], ys[n, ..., idx, c], prob[n, ..., idx, c]])
                imgs = img if imgs is None else np.concatenate((imgs, img), 0)
            img_dict = {'class %d'%c: imgs}
        return img_dict
