# -*- coding: utf-8 -*-
"""
defense model
RenMin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


import recon_fn as RF
from pca_select_model import DAE, Encoder, PCASelection
from arcface_model import Backbone
from light_cnn import LightCNN_29Layers
import pdb



class DefenseModel(nn.Module):
    def __init__(self, config):
        super(DefenseModel, self).__init__()
        #pdb.set_trace()
        self.reg_mod = config.recg_modGet()
        
        self.dae = DAE()
        dae_data = torch.load(config.ckpt_daeGet(), map_location=lambda storage, loc:storage)
        self.dae.load_state_dict(dae_data['model'])
        
        self.encoder = Encoder()
        en_data = torch.load(config.ckpt_encoderGet(), map_location=lambda storage, loc:storage)
        self.encoder.load_state_dict(en_data['model'])
        
        self.pca_layer = PCASelection()
        pca_data = torch.load(config.ckpt_PCAfcGet(), map_location=lambda storage, loc:storage)
        self.pca_layer.load_state_dict(pca_data['model'])
        
        self.recon_fn = RF.EigRecon_Act(config.eigen_faceGet())
        
        if self.reg_mod=='arcface':
            self.reg_model = Backbone(50, 0.4, 'ir_se')
            reg_data = torch.load(config.ckpt_recgGet(), map_location=lambda storage, loc:storage)
            self.reg_model.load_state_dict(reg_data)
        elif self.reg_mod=='lightcnn29':
            self.reg_model = LightCNN_29Layers()
            pre_dict = torch.load(config.ckpt_recgGet(), map_location=lambda storage, loc:storage)
            pre_dict = pre_dict['state_dict']
            pre_dict = {k[7:]:v for k,v in pre_dict.items()}
            self.reg_model.load_state_dict(pre_dict)
        
    def get_pca_act(self, x):
        uni_dis = torch.rand(x.size()).cuda()
        pca_act = x>uni_dis
        return pca_act
    
    def get_pca_act_fix(self, x):
        pca_act = x>x.mean()
        return pca_act

    def proc_before_reg(self, x):
        if self.reg_mod=='arcface':
            x = x.expand(x.size(0),3,x.size(2),x.size(3))
        elif 'lightcnn' in self.reg_mod:
            x = F.interpolate(x, size=[128,128], mode='bilinear')
        return x
    
    def reconstrcution(self, x_raw):
        self.dae.eval()
        self.encoder.eval()
        self.pca_layer.eval()
        
        _, x = self.dae(x_raw)
        x = self.encoder(x)
        x = self.pca_layer(x)
        pca_act = self.get_pca_act(x)
        x_recon = self.recon_fn.recon(x_raw, pca_act)
        return x_recon
        
        
    def forward(self, x_raw):
        #pdb.set_trace()
        self.dae.eval()
        self.encoder.eval()
        self.pca_layer.eval()
        self.reg_model.eval()
            
        _, x = self.dae(x_raw)
        x = self.encoder(x)
        x = self.pca_layer(x)
        pca_act = self.get_pca_act(x)
        x_recon = self.recon_fn.recon(x_raw, pca_act)
        x_recon = self.proc_before_reg(x_recon)
        feat = self.reg_model(x_recon)
        return feat
        
        
