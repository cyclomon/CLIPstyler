import sys
from pathlib import Path
import torch
import torch.optim as optim
from torchvision import transforms, models
import tempfile
import StyleNet
import utils
import clip
import torch.nn.functional as F
from template import imagenet_templates
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
import cog
from argparse import Namespace


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.VGG = models.vgg19(pretrained=True).features
        self.VGG.to(self.device)
        for parameter in self.VGG.parameters():
            parameter.requires_grad_(False)
        self.style_net = StyleNet.UNet()
        self.style_net.to(self.device)
        self.clip_model, preprocess = clip.load('ViT-B/32', self.device, jit=False)

    @cog.input("image", type=Path, help="Input image (will be cropped before style transfer)")
    @cog.input("text", type=str, help="text for style transfer")
    @cog.input("iterations", type=int, default=100, help="training iterations")
    def predict(self, image, text, iterations):
        training_args = {
            "lambda_tv": 2e-3,
            "lambda_patch": 9000,
            "lambda_dir": 500,
            "lambda_c": 150,
            "crop_size": 128,
            "num_crops": 64,
            "img_size": 512,
            "max_step": iterations,
            "lr": 5e-4,
            "thresh": 0.7,
            "content_path": str(image),
            "text": text
        }
        args = Namespace(**training_args)
        out_path = Path(tempfile.mkdtemp()) / "out.png"

        content_path = args.content_path
        content_image = utils.load_image2(content_path, img_size=512)

        content_image = content_image.to(self.device)

        content_features = utils.get_features(img_normalize(content_image, self.device), self.VGG)

        content_weight = args.lambda_c

        optimizer = optim.Adam(self.style_net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        steps = args.max_step

        total_loss_epoch = []
        cropper = transforms.Compose([
            transforms.RandomCrop(args.crop_size)
        ])
        augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224)
        ])
        prompt = args.text
        source = "a Photo"
        with torch.no_grad():
            template_text = compose_text_with_templates(prompt, imagenet_templates)
            tokens = clip.tokenize(template_text).to(self.device)
            text_features = self.clip_model.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            template_source = compose_text_with_templates(source, imagenet_templates)
            tokens_source = clip.tokenize(template_source).to(self.device)
            text_source = self.clip_model.encode_text(tokens_source).detach()
            text_source = text_source.mean(axis=0, keepdim=True)
            text_source /= text_source.norm(dim=-1, keepdim=True)
            source_features = self.clip_model.encode_image(clip_normalize(content_image, self.device))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

        num_crops = args.num_crops
        for epoch in range(0, steps + 1):

            scheduler.step()
            target = self.style_net(content_image, use_sigmoid=True).to(self.device)
            target.requires_grad_(True)

            target_features = utils.get_features(img_normalize(target, self.device), self.VGG)

            content_loss = 0

            content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
            content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

            loss_patch = 0
            img_proc = []
            for n in range(num_crops):
                target_crop = cropper(target)
                target_crop = augment(target_crop)
                img_proc.append(target_crop)

            img_proc = torch.cat(img_proc, dim=0)
            img_aug = img_proc

            image_features = self.clip_model.encode_image(clip_normalize(img_aug, self.device))
            image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

            img_direction = (image_features - source_features)
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

            text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
            loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
            loss_temp[loss_temp < args.thresh] = 0
            loss_patch += loss_temp.mean()

            glob_features = self.clip_model.encode_image(clip_normalize(target, self.device))
            glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

            glob_direction = (glob_features - source_features)
            glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

            loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

            reg_tv = args.lambda_tv * get_image_prior_losses(target)

            total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv + args.lambda_dir * loss_glob
            total_loss_epoch.append(total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == steps:
                yield checkin(epoch, target, total_loss, content_loss, loss_patch, loss_glob, reg_tv, out_path)

        return out_path


@torch.no_grad()
def checkin(epoch, target, total_loss, content_loss, loss_patch, loss_glob, reg_tv, out_path):
    sys.stderr.write(f'After {epoch} iterations')
    sys.stderr.write(f'Total loss: {total_loss.item()}')
    sys.stderr.write(f'Content loss: {content_loss.item()}')
    sys.stderr.write(f'patch loss: {loss_patch.item()}')
    sys.stderr.write(f'dir loss: {loss_glob.item()}')
    sys.stderr.write(f'TV loss: {reg_tv.item()}')
    output_image = target.clone()
    output_image = torch.clamp(output_image, 0, 1)
    output_image = adjust_contrast(output_image, 1.5)
    save_image(output_image, str(out_path), nrow=1, normalize=True)
    return out_path


def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]
