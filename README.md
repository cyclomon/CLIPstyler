# CLIPstyler
## Official Pytorch implementation of "CLIPstyler:Image Style Transfer with a Single Text Condition"

![MAIN3_e2-min](https://user-images.githubusercontent.com/94511035/142139437-9d91f39e-b3d7-46cf-b43b-cb7fdead69a8.png)

### Environment
Pytorch 1.7.1, Python 3.6

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

### Style Transfer with Single-image

To train the model, run

```
python train_CLIPstyler.py --content_path ./content/face.jpg \
--content_name face --exp_name exp1 \
--text "Sketch with black pencil"
```

To change the style of custom image, please change the ```--content_path``` argument

edit the text condition with ```--text``` argument

### Fast Style Transfer
Before training, plase download DIV2K dataset [LINK](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

We recomment to use Training data of High-Resolution(HR) images.

To train the model, please download the pre-trained vgg encoder & decoder models in [LINK](https://drive.google.com/drive/folders/17UDzXtp9IZlerFjGly3QEm2uU3yi7siO?usp=sharing).

Please save the downloaded models in ```./models``` directory

Then, run the command

```
python train_fast.py --content_path $DIV2K_DIR$ \
--name exp1 \
--text "Sketch with black pencil" --test_dir ./test_set
```

Please set the ```$DIV2K_DIR$``` as the directory in which DIV2K images are saved.

To test the fast style transfer model, 

```
python test_fast.py --test_dir ./test_set --decoder ./model_fast/clip_decoder_iter_200.pth.tar
```

Change the argument ```--decoder``` to other trained models for testing on different text conditions.

We provide several fine-tuned decoders for several text conditions. [LINK](https://drive.google.com/drive/folders/1U-4tEigPaJxfXRMnEdRDtyQ99O5ondrs?usp=sharing)

To use high-resolution image, please add ```--hr_dir ./hr_set``` to test command. 
