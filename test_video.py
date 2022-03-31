import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
from torchvision import transforms
from torchvision.utils import save_image
import fast_stylenet
from torchvision.transforms.functional import adjust_contrast

def test_transform(width,height, crop):
    transform_list = []
    if width != 0:
        transform_list.append(transforms.Resize((width,height)))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_path', type=str,
                    help='File path to the content video')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='./model_fast/clip_decoder_sketch_blackpencil.pth.tar')
# Additional options
parser.add_argument('--content_width', type=int, default=0,
                    help='New (minimum) width for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--content_height', type=int, default=0,
                    help='New (minimum) height for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.mp4',
                    help='The extension name of the output video')
parser.add_argument('--output', type=str, default='video_out',
                    help='Directory to save the output video')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok = True, parents = True)

decoder_path = Path(args.decoder)

assert (args.content_path)
if args.content_path:
    content_path = Path(args.content_path)

decoder = fast_stylenet.decoder
vgg = fast_stylenet.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

network = fast_stylenet.Net(vgg, decoder)
network.eval()
network.to(device)

content_tf = test_transform(args.content_width,args.content_height, args.crop)
        
#get video fps & video size
content_video = cv2.VideoCapture(args.content_path)
fps = int(content_video.get(cv2.CAP_PROP_FPS))
content_video_length = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

assert fps != 0, 'Fps is zero, Please enter proper video path'

pbar = tqdm(total = content_video_length)

if content_path.suffix in [".mp4", ".avi", ".mpg"]:

    output_video_path = output_name = output_dir / '{:s}_{:s}{:s}'.format(
                content_path.stem, decoder_path.stem, args.save_ext)
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
    
    while(True):
        ret, content_img = content_video.read()

        if not ret:
            break

        content = content_tf(Image.fromarray(content_img))
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            _,output = network(content)
            output = adjust_contrast(output,1.5)
        
        output = output.cpu()
        output = output.squeeze(0)
        output = np.array(output)*255
        #output = np.uint8(output)
        output = np.transpose(output, (1,2,0))
        output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

        writer.append_data(np.array(output))
        pbar.update(1)
    
    content_video.release()
