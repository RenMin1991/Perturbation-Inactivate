# -*- coding: utf-8 -*-
"""
train pca selection model with policy gradient
RenMin 20190918
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
#from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.autograd import Variable

from config_train import Config
from vision_image_folder import ImageFolder
from pca_select_model import Encoder, PCASelection, DAE
import pca_loss_fn as PLF
import pdb


# parameters
#pdb.set_trace()
config = Config()

EPOCHES = config.epochGet()
BATCH = config.batchGet()
LR_en = config.lr_encoderGet()
LR_pca = config.lr_pcafcGet()
MOMENTUM = config.momentumGet()
NOISE_SCALE = config.noise_scaleGet()


Lamb = config.lamb_sparseGet()
lambda_mean = config.lamb_meanGet()
eigen_path = config.eigen_faceGet()

decay_step = config.decay_stepGet()
decay_rate = config.decay_rateGet()
dae_path = config.ckpt_daeGet()

data_folder = config.data_folderGet()
save_step = config.save_stepGet()
encoder_path = config.encoder_pathGet()
pcafc_path = config.pcafc_pathGet()
    

# define model

encoder = Encoder()
pca_layer = PCASelection()
dae = DAE()

encoder = encoder.cuda()
pca_layer = pca_layer.cuda()
dae_data = torch.load(dae_path, map_location=lambda storage, loc:storage)
dae.load_state_dict(dae_data['model'])
dae = dae.cuda()

# optimizer
params = []
for name, value in encoder.named_parameters():
    params += [{'params':value, 'lr':LR_en}]
for name, value in pca_layer.named_parameters():
    params += [{'params':value, 'lr':LR_pca}]

optimizer = optim.SGD(params=params, lr=LR_en, momentum=MOMENTUM)
lr_sch = optim.lr_scheduler.StepLR(optimizer, decay_step, decay_rate)


# pre-process
transforms = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    
# get data
data_set = ImageFolder(data_folder, transform=transforms)
data_loader = DataLoader(data_set, batch_size=BATCH, shuffle=True)

# noise function
def noise_fn(inputs, scale):
    """
    add Gaussion noise to inputs
    sacle is the Sigmma of Gaission distribution
    """
    noise = torch.randn(inputs.size())
    inputs_scale = ((inputs**2).sum())**0.5
    noise_scale = ((noise**2).sum())**0.5
    noise = noise*scale*inputs_scale / noise_scale
    noise = Variable(noise, requires_grad=False)
    
    inputs_noise = inputs + noise
    
    return inputs_noise


# loss function
mc_loss = PLF.MCLoss(Lamb, eigen_path)


#train
#pdb.set_trace()
dae.eval()
for epoch in range(EPOCHES):
    running_loss = 0.
    encoder.train()
    pca_layer.train()
    for i, data in enumerate(data_loader, 0):
        # input data
        inputs, _, _ = data
        inputs = Variable(inputs)
        inputs_noise = noise_fn(inputs, NOISE_SCALE)
        inputs = inputs.cuda()
        inputs_noise = inputs_noise.cuda()
        
        # zero the grad
        optimizer.zero_grad()
        
        # forward
        _, inputs_recon = dae(inputs_noise)
        hidden = encoder(inputs_recon)
        recon = pca_layer(hidden)
        
        # loss and backward
        loss, mse_loss = mc_loss(recon, inputs, inputs_noise)
        loss.backward()
        optimizer.step()
        
        # print log
        running_loss += float(loss.item())
        if i%1000==999:
            print ('epoch', epoch+1, 'step', i+1, 'loss', running_loss/1000.)
            running_loss = 0.
    lr_sch.step(epoch)
        
    # save model
    if epoch%save_step==save_step-1:
        en_data = dict(
                optimizer = optimizer.state_dict(),
                model = encoder.state_dict(),
                epoches = epoch+1,
                )
        en_name = encoder_path+'_'+str(epoch+1)+'.pth'
        torch.save(en_data, en_name)
        
        pca_data = dict(
                model = pca_layer.state_dict(),
                epoches = epoch+1,
                )
        pca_name = pcafc_path+'_'+str(epoch+1)+'.pth'
        torch.save(pca_data, pca_name)




