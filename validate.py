import cv2
import numpy as np
import torch
import yaml
from models.pix2pix_model import Pix2PixModel


def create_inputs(images):
    mean = [0.3258, 0.3761, 0.3852]  # BGR values
    std = [0.0280, 0.0275, 0.0372]  # BGR values
    inputs = []
    for img in images:
        img = cv2.resize(img, (128, 96)).astype('float32')
        img = (img/255. - mean)/std
        inputs.append(img)
    return np.concatenate(inputs, axis=-1)


if __name__ == '__main__':
    with open("checkpoints/panoGAN14/options.yml", 'r') as f:
        opt = yaml.load(f, Loader=yaml.Loader)

    model = Pix2PixModel(opt)
    model.load_networks(75)

    x = []
    for img in ['4_15.png', '4_16.png', '4_17.png']:
        f = cv2.imread(img)
        x.append(f)
    x = create_inputs(x)
    x = np.expand_dims(x, axis=0)
    x = torch.Tensor(x.transpose(0, 3, 1, 2))
    y = model.netG(x)
    print(y.shape)
    y = y.cpu().detach().numpy()[0]
    y = y.transpose(1, 2, 0)
    print(y.shape)

    out = np.zeros((96, 128*(opt.output_nc//3), 3), dtype='float32')
    for i in range(opt.output_nc//3):
        out[:, 128*i:128*(i+1)] = y[:, :, 3*i:3*(i+1)]
    out = (out + 1)/2.0 * 255.0
    out = out.astype('uint8')
    cv2.imwrite('out0.jpg', out)
