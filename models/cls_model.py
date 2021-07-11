import numpy as np
import tensorflow as tf
from abc import ABCMeta,abstractmethod

from utils import eval_methods as EM
from utils import loss_tf as LF
from utils import util as U
from utils.process_methods import one_hot
from models.model import Model


class ClsModel(Model):
    def __init__(self, net, x_suffix, y_suffix, m_suffix=None, dropout=0):
        super().__init__(net)
        self._x_suffix = x_suffix
        self._y_suffix = y_suffix
        self._m_suffix = m_suffix

        self.dropout = dropout

    def get_grads(self, data_dict):
        xs = data_dict[self._x_suffix]

        with tf.GradientTape() as tape:
            logits = self.net(xs, self.dropout, True)
            loss = self._get_loss(logits, data_dict)                       
        grads = tape.gradient(loss, self.net.trainable_variables)
        return grads

    def eval(self, data_dict, **kwargs):
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix][..., 2:3]
        
        
        logits = self.net(xs, 0, False)

        loss = self._get_loss(logits, data_dict)
        mse = EM.mean_squared_error(logits, ys)

        eval_results = {'loss': loss,
                        'mse': mse}

        need_preds = kwargs.get('need_preds', False)
        if need_preds:
            eval_results.update({'pred': logits})

        need_all_info = kwargs.get('need_all_info', False)
        if need_all_info:
            all_info = data_dict[self._y_suffix][..., :3]
            eval_results.update({'all_info': all_info})

        mc_eval = kwargs.get('mc_eval', None)
        if mc_eval is not None:
            eval_results.update({'mc': self._mc_eval(xs, mc_eval)})

        return eval_results

    def predict(self, data_dict):
        return self.net(data_dict[self._x_suffix])

    def _get_loss(self, logits, data_dict):
        ys = data_dict[self._y_suffix][..., 2:3]
        loss = tf.losses.mse(logits, ys)
        return loss


    def _mc_eval(self, xs, mc_setting):
        t_stochastic = mc_setting['t_stochastic']
        dropout = mc_setting['dropout']
        ages = np.zeros((xs.shape[0], t_stochastic))
        for i in range(t_stochastic):
            logits = self.net(xs, dropout, False)
            ages[..., i:i+1] = logits
        return ages