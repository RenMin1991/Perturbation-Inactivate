# -*- coding: utf-8 -*-
"""
configuration of AID
RenMin
"""

class Config(object):
    def __init__(self):
        self.recg_mod = 'arcface'
        
        self.ckpt_dae = 'checkpoint/dae2_200.pth'
        self.ckpt_encoder = 'checkpoint/encoder_lfw.pth'
        self.ckpt_PCAfc = 'checkpoint/pca_layer_lfw.pth'
        self.ckpt_recg = 'checkpoint/irse_50.pth'
        
        self.eigen_face = 'face_eigen/lfw_eig.pth'
        
        self.data_folder = '../data/face/facescrub_images/'
        self.feat_dim = 512
        self.feat_path = 'featurs/feat.pth'


    def recg_modGet(self):
        return self.recg_mod
    def ckpt_daeGet(self):
        return self.ckpt_dae
    def ckpt_encoderGet(self):
        return self.ckpt_encoder
    def ckpt_PCAfcGet(self):
        return self.ckpt_PCAfc
    def ckpt_recgGet(self):
        return self.ckpt_recg
    def eigen_faceGet(self):
        return self.eigen_face
    def feat_dimGet(self):
        return self.feat_dim
    def feat_pathGet(self):
        return self.feat_path
    def data_folderGet(self):
        return self.data_folder