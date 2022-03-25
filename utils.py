import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def load_image(img_path, img_size=None):
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # this is from ImageNet dataset
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image
def load_image2(img_path, img_height=None,img_width =None):
    
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image

def im_convert2(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    image = image.transpose(1,2,0)
       # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image
def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features



def rand_bbox(size, res):
    W = size
    H = size
    cut_w = res
    cut_h = res
    tx = np.random.randint(0,W-cut_w)
    ty = np.random.randint(0,H-cut_h)
    bbx1 = tx
    bby1 = ty
    return bbx1, bby1


def rand_sampling(args,content_image):
    bbxl=[]
    bbyl=[]
    bbx1, bby1 = rand_bbox(args.img_size, args.crop_size)
    crop_img = content_image[:,:,bby1:bby1+args.crop_size,bbx1:bbx1+args.crop_size]
    return crop_img

def rand_sampling_all(args):
    bbxl=[]
    bbyl=[]
    out = []
    for cc in range(50):
        bbx1, bby1 = rand_bbox(args.img_size, args.crop_size)
        bbxl.append(bbx1)
        bbyl.append(bby1)
    return bbxl,bbyl
