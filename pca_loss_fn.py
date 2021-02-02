# -*- coding: utf-8 -*-
"""
Loss functions for PCA selection model
RenMin 20190918
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

   
class MCLoss(nn.Module):
    def __init__(self, lamb, eigen_path, num_sample=30):
        super(MCLoss, self).__init__()
        self.eigen_data = torch.load(eigen_path)
        self.lamb = lamb
        self.num_sample = num_sample
        
    def MCSampling(self, pca_act):
        num_eig = pca_act.size(0)
        prob = pca_act.detach().cpu().unsqueeze(0).expand(self.num_sample, num_eig)
        uni_dis = torch.rand(prob.size())
        samples = prob>uni_dis
        return samples
    
    def Loss(self, samples, image, image_noise):
        #pdb.set_trace()
        index_samples = np.where(samples.numpy()==1)
        num_eig = index_samples[1].max() + 1
        samples = samples[:,:num_eig]
        eig_face = self.eigen_data['eig_vec'][:,:num_eig]
        avg_face = self.eigen_data['avg_face'].cuda()
        _, H, W = image.size()
        D, _ = eig_face.size()
        image = image.view(1, H*W).t() # 1 X H*W
        image = image - avg_face.unsqueeze(1) # subtraction by average face
        image_noise = image_noise.view(1, H*W).t() # 1 X H*W
        image_noise = image_noise - avg_face.unsqueeze(1) # subtraction by average face
        
        #reconstruction
        samples_exp = samples.unsqueeze(1).expand(self.num_sample, D, num_eig)
        eig_face = eig_face.unsqueeze(0).expand(self.num_sample, D, num_eig)
        eig_face = samples_exp.float() * eig_face
        eig_face = eig_face.cuda()
        
        recon_samples = torch.matmul(eig_face,torch.matmul(eig_face.permute(0,2,1), image_noise)) # num_sample X D X 1
        
        recon_loss = ((recon_samples.squeeze(2) - image.t().expand(self.num_sample,D))**2).mean(1)
        regular = samples.sum(1).float().cuda()
        
        loss = recon_loss + self.lamb * regular
        return loss
        
    def LogLikehood(self, pca_act, samples):
        pca_act = pca_act.unsqueeze(0).expand(self.num_sample, samples.size(1))
        samples = samples.float().cuda()
        probability = ((1-samples) - pca_act).abs()
        Likehood = torch.log(probability[:,0])
        for i in range(samples.size(1)-1):
            Likehood = Likehood + torch.log(probability[:,i+1])
        #pdb.set_trace()
        #LogLikehood = torch.log(Likehood)
        return Likehood
        
        
    def forward(self, pca_act, images, images_noise):
        #pdb.set_trace()
        losses = torch.zeros(self.num_sample, images.size(0)).cuda()
        likehoods = torch.zeros(self.num_sample, images.size(0)).cuda()
        for i in range(images.size(0)):
            samples = self.MCSampling(pca_act[i]) # num_sample X num_eig
            losses[:,i] = self.Loss(samples, images[i], images_noise[i]) # num_sample
            likehoods[:,i] = self.LogLikehood(pca_act[i], samples) # num_sample
        
        loss_mean = losses.mean()
        losses = losses - loss_mean
        mc_loss = (losses * likehoods).sum(0).mean()
        return mc_loss, loss_mean






















      
