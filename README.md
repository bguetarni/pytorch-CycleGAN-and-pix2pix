To view training results and loss plots run `python -m visdom.server` and go to http://localhost:8097.

To train a model:

```
bash python train.py --dataroot /data/panorama/ --model pix2pix --name panorama --dataset_mode custom --netG resnet_9blocks --input_nc 6 --output_nc 6 --display_ncols 0 --batch_size 8 --gan_mode wgangp --lambda_L1 150 --lambda_SSIM 200 --lr_policy plateau --gpu_ids 0
 ```

To see more intermediate results, check out `./checkpoints/facades_pix2pix/web/index.html`

## Some training results

| Epoch | input | ground-truth | output |
|-------|-------|--------------|--------|
| 01 | ![](./imgs/intermediate_results/epoch001_real_A.png) | ![](./imgs/intermediate_results/epoch001_real_B.png) | ![](./imgs/intermediate_results/epoch001_fake_B.png) |
| 50 | ![](./imgs/intermediate_results/epoch050_real_A.png) | ![](./imgs/intermediate_results/epoch050_real_B.png) | ![](./imgs/intermediate_results/epoch050_fake_B.png) |
| 100 | ![](./imgs/intermediate_results/epoch100_real_A.png) | ![](./imgs/intermediate_results/epoch100_real_B.png) | ![](./imgs/intermediate_results/epoch100_fake_B.png) |
| 150 | ![](./imgs/intermediate_results/epoch150_real_A.png) | ![](./imgs/intermediate_results/epoch150_real_B.png) | ![](./imgs/intermediate_results/epoch150_fake_B.png) |

To see the last training curves copy the files from the **visdom/** folder to **~/.visdom/**. Then run `visdom -en_path ~/.visdom/`.
