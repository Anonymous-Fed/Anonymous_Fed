
import gfedplat as fp
import numpy as np
import torch
class FedAvg(fp.Algorithm):
    def __init__(self,
                 name='FedAvg',
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
                 *args,
                 **kwargs):
        
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
    def run(self):
        
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            
            m_locals, _ = self.train()
            
            self.weight_aggregate(m_locals)
            
            self.current_training_num += self.epochs * batch_num
