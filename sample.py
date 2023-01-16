import json

import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow

device = torch.device("cuda")

output_folder = 'output/'
model_name = 'glow_checkpoint_25000.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
    
image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
# image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model_data = torch.load(output_folder + model_name)
model.load_state_dict(model_data["model"])
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

def sample(model, n=30):
    with torch.no_grad():
        x = model(y_onehot=None, temperature=1e-1, reverse=True)
        x = x[:n]
        images = postprocess(x)

    return images.cpu()

images = sample(model)
path = output_folder + 'glow-samples.png'
save_image(images, path, nrow=6)