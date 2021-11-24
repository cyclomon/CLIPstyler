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

def train_transform(crop_size=224):
    transform_list = [
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
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

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

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

def load_image(img_path, img_size=None):
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def clip_normalize(image):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def reverse_normalize(image):
    mean=torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).to(device)
    std=torch.tensor([1./0.229, 1./0.224, 1./0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image
def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default ='/mnt/F/DIV2K_RW/DIV2K_Realistic_Wild/Train/HR')
parser.add_argument('--test_dir', type=str, default ='./test_set') 
parser.add_argument('--hr_dir', type=str)  
parser.add_argument('--img_dir', type=str, default ='./test_set')     
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./model_fast',
                    help='Directory to save the model')

parser.add_argument('--text', default='Fire',
                    help='text condition')
parser.add_argument('--name', default='none',
                    help='name')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--clip_weight', type=float, default=10.0)
parser.add_argument('--tv_weight', type=float, default=1e-4)
parser.add_argument('--glob_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--num_test', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=200)
parser.add_argument('--save_img_interval', type=int, default=100)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--thresh', type=float, default=0.7)
parser.add_argument('--decoder', type=str, default='./models/decoder.pth')
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

decoder = fast_stylenet.decoder
vgg = fast_stylenet.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

decoder.load_state_dict(torch.load(args.decoder))

network = fast_stylenet.Net(vgg, decoder)
network.train()
network.to(device)
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

source = "a Photo"

with torch.no_grad():
    prompt = args.text
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    
    
content_tf = train_transform(args.crop_size)
hr_tf = hr_transform()
test_tf = test_transform()


content_dataset = FlatFolderDataset(args.content_dir, content_tf)
test_dataset = FlatFolderDataset(args.test_dir, test_tf)


augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
test_iter = iter(data.DataLoader(
    test_dataset, batch_size=args.num_test,
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
test_images1 = next(test_iter)
test_images1 = test_images1.cuda()

if args.hr_dir is not None:
    hr_dataset = FlatFolderDataset(args.hr_dir, hr_tf)
    
    hr_iter = iter(data.DataLoader(
        hr_dataset, batch_size=1,
        num_workers=args.n_threads))
    hr_images = next(hr_iter)
    hr_images = hr_images.cuda()

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    loss_c, out_img = network(content_images)
    loss_patch = 0
    aug_img = []
    for it in range(16):
        out_aug = augment_trans(out_img)
        aug_img.append(out_aug)
    aug_img = torch.cat(aug_img,dim=0)
    
    source_features = clip_model.encode_image(clip_normalize(content_images))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
    
    image_features = clip_model.encode_image(clip_normalize(aug_img))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    
    img_direction = (image_features-source_features.repeat(16,1))
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
    
    text_direction = (text_features-text_source)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction.repeat(image_features.size(0),1), dim=1))#.mean()

    loss_temp[loss_temp<args.thresh] =0

    loss_patch+=loss_temp.mean()
    glob_features = clip_model.encode_image(clip_normalize(out_img))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    
    glob_direction = (glob_features-source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    
    loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction.repeat(glob_features.size(0),1), dim=1)).mean()
    
    loss_c = args.content_weight * loss_c
    reg_tv = args.tv_weight*get_image_prior_losses(out_img)
    
    loss = loss_c + args.clip_weight*loss_patch + args.glob_weight*loss_glob + reg_tv

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%10==0:
        print('loss_content:' + str(loss_c.item()))
        print('loss_patch:' + str(loss_patch.item()))   
        print('loss_dir:' + str(loss_glob.item())) 
        print('loss_tv:' + str(reg_tv.item()))
        
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = fast_stylenet.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'clip_decoder_iter_{:d}.pth.tar'.format(i + 1))

    if (i + 1) % args.save_img_interval ==0 :
        with torch.no_grad():
            _, test_out1 = network( test_images1)
            test_out1 = adjust_contrast(test_out1,1.5)
            output_test = torch.cat([test_images1,test_out1],dim=0)

            output_name = './output_fast/test1_'+ args.text +'_'+ str(i+1)+'.png'
            save_image(output_test, str(output_name),nrow=test_out1.size(0),normalize=True,scale_each=True)
            
            if args.hr_dir is not None:
                _, test_out = network(hr_images)
                test_out = adjust_contrast(test_out,1.5)
                output_name = './output_fast/hr_'+ args.name+'_'+ args.text +'_'+ str(i+1)+'.png'
                save_image(test_out, str(output_name),nrow=test_out.size(0),normalize=True,scale_each=True)
            
