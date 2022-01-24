import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from template import imagenet_templates
import fast_stylenet
from sampler import InfiniteSamplerWrapper
import clip
from template import imagenet_templates
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time

def intp(model1, model2,model3, decay=0.999):

    for p_out, p_in1, p_in2 in zip(model3.parameters(), model1.parameters(), model2.parameters()):
            p_out.data = nn.Parameter(p_in1*(1-decay) +p_in2*(decay));
    return model3

def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def hr_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, default ='./test_set') 
parser.add_argument('--hr_dir', type=str)     
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
# training options
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--num_test', type=int, default=16)

parser.add_argument('--decoder_src', type=str, default='./model_fast/clip_decoder_stonewall.pth.tar')

parser.add_argument('--decoder_trg', type=str, default='./model_fast/clip_decoder_desert.pth.tar')
args = parser.parse_args()

device = torch.device('cuda')

decoder = fast_stylenet.decoder_cls()
decoder2 = fast_stylenet.decoder_cls()
decoder_out = fast_stylenet.decoder_cls()

vgg = fast_stylenet.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])


decoder.decode.load_state_dict(torch.load(args.decoder_src))
decoder2.decode.load_state_dict(torch.load(args.decoder_trg))

vgg = vgg.to(device)
decoder = decoder.to(device)
decoder2 =decoder2.to(device)
decoder_out =decoder_out.to(device)

test_tf = test_transform()
test_dataset = FlatFolderDataset(args.test_dir, test_tf)
test_iter = iter(data.DataLoader(
    test_dataset, batch_size=args.num_test,
    num_workers=args.n_threads))

test_images1 = next(test_iter)
test_images1 = test_images1.cuda()

if args.hr_dir is not None:
    hr_tf = hr_transform()
    hr_dataset = FlatFolderDataset(args.hr_dir, hr_tf)
    hr_iter = iter(data.DataLoader(
    hr_dataset, batch_size=1,
    num_workers=args.n_threads))

    hr_images = next(hr_iter)
    hr_images = hr_images.cuda()

with torch.no_grad():
    outs = [test_images1]
    for i in range(6):
        decoder_intp = intp(decoder,decoder2,decoder_out,i*0.2)
        enc_feat = vgg(test_images1)
        test_out1 = decoder_intp(enc_feat)
        
        test_out1 = adjust_contrast(test_out1,1.5)
        outs.append(test_out1)
        
    output_test = torch.cat(outs,dim=0)
    output_name = './output_test/test_intp.png'
    save_image(output_test, str(output_name),nrow=test_out1.size(0),normalize=True,scale_each=True)
    

            
