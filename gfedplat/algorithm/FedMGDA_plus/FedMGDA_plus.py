
import gfedplat as fp
import torch
import numpy as np
import copy
from gfedplat.algorithm.common.utils import get_d_mgdaplus_d
import time
class FedMGDA_plus(fp.Algorithm):
    def __init__(self,
                 name='FedMGDA+',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 params=None,
                 epsilon=0.1,  
                 *args,
                 **kwargs):
        if params is not None:
            epsilon = params['epsilon']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' epsilon' + str(epsilon)
        
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.epsilon = epsilon
        
        self.comm_log['d_optimality_history'] = []  
        self.comm_log['d_descent_history'] = []  
    def run(self):
        
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        
        while not self.terminated():
            
            m_locals, l_locals = self.train()
            
            g_locals = []
            old_models = self.module.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                grad = old_models - m_locals[idx].span_model_params_to_vec()  
                g_locals.append(grad)
            g_locals = torch.stack(g_locals)  
            g_locals /= torch.norm(g_locals, dim=1).reshape(-1, 1)
            training_nums = self.get_clinet_attr('local_training_number')
            lambda0 = np.array(training_nums) / sum(training_nums)
            
            d, d_optimal_flag, d_descent_flag = get_d_mgdaplus_d(g_locals, self.device, self.epsilon, lambda0)
            
            self.update_module(self.module, self.optimizer, self.lr, d)
            
            self.current_training_num += self.epochs * batch_num
            
            self.comm_log['d_optimality_history'].append(d_optimal_flag)
            self.comm_log['d_descent_history'].append(d_descent_flag)
