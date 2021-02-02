# -*- coding: utf-8 -*-
"""
functions of reconstruction of face by EigenFace and KNN
RenMin 20190910
"""

import torch
import torch.nn.functional as F

import pdb


class EigRecon_Act(object):
    """
    reconstruction image according to the
    activited nodes
    """
    
    def __init__(self, eigen_path):
        self.eigen_data = torch.load(eigen_path)
        self.eig_face = self.eigen_data['eig_vec'][:,:2500].cuda()
        self.avg_face = self.eigen_data['avg_face'].cuda()
        
    def recon(self, x, act):
        # x:   B X 1 X H X W
        # act: 2500
        pdb.set_trace()
        B, _, H, W = x.size()
        x = x.view(B, H*W).t()
        x = x - self.avg_face.unsqueeze(1).expand(H*W, B) # substract the mean vector
        x_recon = torch.zeros(x.size()).cuda()
        
        for i in range(B):
            x_sub = x[:, i]
            act_sub = act[i, :]
            index_sub = (act_sub.squeeze())>0
            eig_v = self.eig_face[:, index_sub]
            x_recon_sub = torch.mm(eig_v, torch.mm(eig_v.t(), x_sub.unsqueeze(1)))
            x_recon_sub = x_recon_sub + self.avg_face.unsqueeze(1)
            x_recon[:, i] = x_recon_sub.squeeze(1)
            
        x_recon = x_recon.t().view(B, 1, H, W)
        return x_recon
        


















