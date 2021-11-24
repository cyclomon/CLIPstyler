#!/usr/bin/env bash

python train_CLIPstyler.py --content_path ./content/con3.jpg \
--content_name con3 --exp_name exp_p128_1 \
--text "A painting of sunflowers" --max_step 200 --lambda_tv 2e-3 \
--lambda_sty 9000 --lambda_glob 500 \
--crop_size 128 --thresh 0.7 --lr 5e-4
