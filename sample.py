import json

import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, postprocess as postprocess
from model import Glow

device = torch.device("cuda")

# output_folder = 'glow/'
# model_name = 'glow_affine_coupling.pt'
output_folder = 'output/'
model_name = 'glow_checkpoint_7000.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
    
image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
# image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model_data = torch.load(output_folder + model_name)
model.load_state_dict(model_data["model"])
# model.load_state_dict(model_data)
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

def sample(model, n=30):
    with torch.no_grad():
        x = model(y_onehot=None, temperature=1., reverse=True)
        print(x.mean())
        x = x[:n]
        images = postprocess(x)
        # images = x

    return images.cpu()

images = sample(model)
# print(images.mean())
# print(images.shape)

# images = torch.tensor(test_cifar.data)[:30].permute(0,3,1,2)
# print(images[0,0])
# print(samples[0,0])

path = output_folder + 'glow-samples.png'
grid = make_grid(images[:30], nrow=6).permute(1,2,0)
plt.figure(figsize=(10,10))
# plt.imshow(images[0].permute(1,2,0)[:,:,1])
plt.imshow(grid)
plt.axis('off')
plt.savefig(path)