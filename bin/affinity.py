"""
Written by Jonghwan Choi at 1 Apr 2022
https://github.com/mathcom/MTMR
"""
import torch
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.encoders import *
from rdkit import Chem


class affinity(object):
    def __init__(self, target_protein_sequence, device=None):
        super(affinity, self).__init__()
        self.target = target_protein_sequence
        ## Torch setting
        if device is None:
            self.device = torch.device('cpu') if device is None else device
        ## DeepPurpose
        self.model = models.model_pretrained(model = 'MPNN_CNN_BindingDB')
        self.model.device = device
        self.model.model_drug.to(torch.float64)
        self.model.model_protein.to(torch.float64)
        self.model.model.to(torch.float64)
        
    def __call__(self, s):
        if s is None:
            return 0.
        elif Chem.MolFromSmiles(s) is None:
            return 0.
        else:
            df_data = data_process_repurpose_virtual_screening([s],
                                                               [self.target],
                                                               self.model.drug_encoding,
                                                               self.model.target_encoding,
                                                               'virtual screening')
            y_pred = self.model.predict(df_data)
            return y_pred[0]