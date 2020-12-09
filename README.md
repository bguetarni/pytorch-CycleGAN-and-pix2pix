
To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

Train a model:
```bash
python train.py --dataroot dataset.npz --model pix2pix --name panorama --dataset_mode custom --netG resnet_6blocks --input_nc 45 --output_nc 18 --display_ncols 0 --batch_size 8

```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`
