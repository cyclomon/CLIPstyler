# CLIPstyler
## Official source code of "CLIPstyler:Image Style Transfer with a Single Text Condition"

![MAIN3_e2-min](https://user-images.githubusercontent.com/94511035/142139437-9d91f39e-b3d7-46cf-b43b-cb7fdead69a8.png)

### Style Transfer with Single-image
For simple experiment, please run
```
bash run_CLIPstyler.sh
```

To change the settings, edit the command 
```
python train_CLIPstyler.py --content_path ./content/face.jpg \
--content_name face --exp_name p128_g500_s9000_th07 \
--text "Fire"
```

edit the text condition with ```--text``` argument

### Fast Style Transfer
