# -*- coding: utf-8 -*-
"""
configuration of AID
RenMin
"""

class Config(object):
    def __init__(self):
        self.epoch = 200
        self.batch = 20
        self.lr_encoder = 1e-2
        self.lr_pcafc = 1e-2
        self.momentum = 0.9
        self.decay_step = 20
        self.decay_rate = 0.5
        
        self.noise_scale = 0.04
        self.lamb_sparse = 0.015
        self.lamb_mean = 1.
        
        self.data_folder = '../data/face/facescrub_images/'
        self.ckpt_dae = 'checkpoint/dae2_200.pth'
        self.eigen_face = 'face_eigen/lfw_eig.pth'
        
        self.save_step = 1
        self.encoder_path = 'checkpoint/encoder'
        self.pcafc_path = 'checkpoint/pca_fc'


    def epochGet(self):
        return self.epoch
    def batchGet(self):
        return self.batch
    def lr_encoderGet(self):
        return self.lr_encoder
    def lr_pcafcGet(self):
        return self.lr_pcafc
    def momentumGet(self):
        return self.momentum
    def decay_stepGet(self):
        return self.decay_step
    def decay_rateGet(self):
        return self.decay_rate
    def noise_scaleGet(self):
        return self.noise_scale
    def lamb_sparseGet(self):
        return self.lamb_sparse
    def lamb_meanGet(self):
        return self.lamb_mean
    def data_folderGet(self):
        return self.data_folder
    def ckpt_daeGet(self):
        return self.ckpt_dae
    def save_stepGet(self):
        return self.save_step
    def encoder_pathGet(self):
        return self.encoder_path
    def pcafc_pathGet(self):
        return self.pcafc_path
    def eigen_faceGet(self):
        return self.eigen_face