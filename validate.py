import cv2
import os
import argparse
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
    with open("checkpoints/panoGAN10/options.yml", 'r') as f:
        opt = yaml.load(f, Loader=yaml.Loader)
    # region
    """
    opt = argparse.Namespace()
    opt.batch_size = 12
    opt.beta1 = 0.99
    opt.checkpoints_dir = './checkpoints'
    opt.continue_train = False
    opt.crop_size = 256
    opt.dataroot = '/data/panorama/'
    opt.dataset_mode = 'custom'
    opt.direction = 'AtoB'
    opt.display_env = 'panoGAN8'
    opt.display_freq = 400
    opt.display_id = 1
    opt.display_ncols = 0
    opt.display_port = 8097
    opt.display_server = 'http://localhost'
    opt.display_winsize = 256
    opt.epoch = 'latest'
    opt.epoch_count = 1
    opt.gan_mode = 'lsgan'
    opt.gpu_ids = [0]
    opt.init_gain = 0.02
    opt.init_type = 'normal'
    opt.input_nc = 6
    opt.isTrain = True
    opt.lambda_L1 = 10.0
    opt.lambda_SSIM = 300.0
    opt.load_iter = 0
    opt.load_size = 286
    opt.lr = 0.0001
    opt.lr_decay_iters = 20
    opt.lr_policy = 'step'
    opt.max_dataset_size = 'inf'
    opt.model = 'pix2pix'
    opt.n_epochs = 10
    opt.n_epochs_decay = 150
    opt.n_layers_D = 3
    opt.name = 'panorama'
    opt.ndf = 64
    opt.netD = 'basic'
    opt.netG = 'resnet_9blocks'
    opt.ngf = 64
    opt.no_dropout = False
    opt.no_flip = False
    opt.no_html = False
    opt.norm = 'batch'
    opt.num_threads = 4
    opt.output_nc = 6
    opt.phase = 'test'
    opt.pool_size = 0
    opt.preprocess = 'resize_and_crop'
    opt.print_freq = 10000
    opt.save_by_iter = False
    opt.save_epoch_freq = 5
    opt.save_latest_freq = 5000
    opt.serial_batches = False
    opt.suffix = ''
    opt.update_html_freq = 1000
    opt.verbose = False
    """
    # endregion

    model = Pix2PixModel(opt)
    model.load_networks(5)

    x = []
    for img in ['2_14.png', '2_15.png', '2_16.png']:
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

    out = np.zeros((96, 128*2, 3), dtype='float32')
    for i in range(2):
        out[:, 128*i:128*(i+1)] = y[:, :, 3*i:3*(i+1)]
    out = (out + 1)/2.0 * 255.0
    out = out.astype('uint8')
    cv2.imwrite('out0.jpg', out)
