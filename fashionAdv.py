import os
import torch
import numpy as np
import random
import pickle
import sys
from tqdm import tqdm
from pytorch_msssim import ms_ssim

from torchvision.utils import save_image
from torch import optim

import functional_tensor as ft

from Instance_Segmentation_Attack.util import *
from Instance_Segmentation_Attack.textural_loss import *
from Instance_Segmentation_Attack.jpeg_compression import jpeg_approximation
from Instance_Segmentation_Attack.image_manipulation import apply_gaussian_filter, adjust_hue

sys.path.append('yolact_code')

from yolact_code.eval import *
from yolact_code.data import *
from yolact_code.layers.modules import MultiBoxLoss


device = torch.device("cuda")
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from kornia import kornia
from kornia.augmentation import random_generator as rg
from kornia.losses import TotalVariation

TV = TotalVariation()

set_cfg("yolact_base_config")

dataset = COCODetection('/content/yolact/data/coco/val2017/', '/content/yolact/data/coco/annotations/instances_val2017.json',
                        transform=NoneTransform(), has_gt=True)

prep_coco_cats()

yolact = Yolact()
yolact.to(device)
yolact.load_weights("/content/yolact/weights/yolact_base_54_800000.pth")
yolact.eval()

for param in yolact.parameters():
    param.requires_grad = False

args = parse_args('')

args.top_k = 5
cfg.mask_proto_debug = args.mask_proto_debug
yolact.detect.use_fast_nms = True
yolact.detect.use_cross_class_nms = False
cfg.train_masks = True
cfg.use_semantic_segmentation_loss = True
cfg.mask_proto_loss = 'l1'
yolact.change_pred_outs(False)


criterion = MultiBoxLoss(num_classes=cfg.num_classes, pos_threshold=cfg.positive_iou_threshold,
                         neg_threshold=cfg.negative_iou_threshold, negpos_ratio=cfg.ohem_negpos_ratio)

conv_id_coco_dset = {dataset.ids[i]: i for i in range(len(dataset))}

file_id = open('/home/smg/v-marc/code/Instance_Segmentation_Attack/data/image_id_list.pkl', 'rb')
imgs_to_attack_cocoid = pickle.load(file_id)
file_id.close()

imgs_to_attack = [conv_id_coco_dset[i] for i in imgs_to_attack_cocoid]


def optimze_attack(optim, img_to_optimize, mask, real_image, target, masks, num_crowd, Crosstyle_texture,
                   content_texture, option):

    optim.zero_grad()
    adv_x = torch.mul((1 - mask), real_image) + torch.mul(mask, img_to_optimize)

    # TV loss
    tv_loss = option['tv_loss_weight'] * TV(adv_x)
    total_loss = tv_loss

    # SSIM loss
    ssim_loss = option['ssim_loss_weight'] * ms_ssim(undo_normalize(real_image), undo_normalize(adv_x), data_range=1,
                                                     size_average=True)
    total_loss -= ssim_loss

    # Texture loss
    Cross_out = zip(vgg(adv_x, style_layers[:4]), vgg(adv_x, style_layers[1:]))
    content_out = vgg(adv_x, content_layers)
    layer_losses = [option['weights'][a] * loss_fns[a](A, B, Crosstyle_texture[a]) / CrossGramMatrix()(A, B).std()
                    for a, (A, B) in enumerate(Cross_out)]
    content_losses = [option['weights'][4 + a] * loss_fns[4 + a](A, content_texture[a]) for a, A in
                      enumerate(content_out)]
    texture_loss = sum(layer_losses) + sum(content_losses)
    total_loss += texture_loss

    # Transform operation
    if random.random() > (1 - option['apply_transform']):
        params = rg.random_perspective_generator(1, 550, 550, 1.0, option['transform'])
        adv_x = undo_normalize(adv_x)

        adv_x = normalize(kornia.apply_perspective(adv_x, params))
        masks = kornia.apply_perspective(masks, params).to(device).squeeze(0)

    if random.random() > (1 - option['apply_gaussian']):  # apply Gaussian Blur
        adv_x = undo_normalize(adv_x)
        adv_x = normalize(apply_gaussian_filter(adv_x, option))

    if random.random() > (1 - option['apply_color_manipulation']):
        adv_x = undo_normalize(adv_x).squeeze(0)
        adv_x = ft.adjust_brightness(adv_x, random.uniform(*option['brightness']))
        adv_x = ft.adjust_saturation(adv_x, random.uniform(*option['saturation']))
        adv_x = ft.adjust_contrast(adv_x, random.uniform(*option['contrast']))
        adv_x = adjust_hue(adv_x, random.uniform(*option['hue']))
        noise = torch.cuda.FloatTensor(adv_x[0].shape).uniform_(-1, 1) * random.uniform(*option['noise'])
        adv_x = normalize((adv_x + noise).unsqueeze(0))

    if random.random() > (1 - option['apply_jpeg']):
        adv_x = undo_normalize(adv_x).squeeze(0)
        adv_x = torch.clamp(adv_x, 0, 1) * 255

        qf = random.uniform(*option['quality_factor'])
        adv_x = jpeg_approximation(adv_x.permute(1, 2, 0).unsqueeze(0), factor=qf) / 255
        adv_x = adv_x.permute(0, 3, 1, 2)
        adv_x = normalize(adv_x).float()

    # Adversarial Loss
    yolact.change_pred_outs(True)
    yolact_loss = criterion(yolact, yolact(adv_x), [target], [masks], [num_crowd])

    adv_loss = option['adv_loss_weight'] * (yolact_loss['S'] * option['yolact_weights']['segmentation_weight'] +
                                            yolact_loss['C'] * option['yolact_weights']['class_weight'])

    total_loss += adv_loss
    total_loss.backward()
    optim.step()


def fashionAdv_attack(content_index_image, setup):
    """

    :param content_index_image:
    :param setup:
    :return:
    """

    # Getting the clean image from dataset.
    content_image, _, _, num_crowds = get_item(content_index_image, dataset)
    if content_index_image == 4011:
        num_crowds = 0
    content_image = content_image.to(device)

    # Generating the ground truth for YOLACT
    yolact.change_pred_outs(False)
    y_hat = yolact(content_image)
    nb_keep = max(1, sum(y_hat[0]['detection']['score'] > setup['threshold_yolact']))
    ind = (y_hat[0]['detection']['class'] != 0).nonzero()[:nb_keep]
    classe = y_hat[0]['detection']['class'][ind]
    box = y_hat[0]['detection']['box'][ind].squeeze(1)
    targets_image = torch.cat([box, classe.float()], dim=1).to(device)
    masks_image = postprocess(y_hat, 550, 550, crop_masks=True, score_threshold=0)[3][ind].squeeze(1).to(device)

    # Open the mask for the attack, genereting from the human parsing
    image_mask = Image.open(f'/home/smg/v-marc/data/mask_upper_shirt/mask_{dataset.ids[content_index_image]:012d}.png')
    content_mask = transforms.ToTensor()(image_mask).to(device)

    # Create the patch
    patch = torch.randn(content_image.size()).type_as(content_image.data).to(device)
    patch.data.copy_(content_image)
    patch = Variable((content_mask * patch), requires_grad=True)

    optimizer = optim.Adam([patch], lr=setup['lr'], amsgrad=True)

    # Open the style image
    im = Image.open(f'/home/smg/v-marc/data/fashion_pattern/{setup["Texture_style"]:0>2d}.jpg')
    style_image = transforms.ToTensor()(im)
    style_image = normalize(style_image).cuda()

    style_weights = [setup['textural_loss_weight'] / n ** 2 for n in [64, 128, 256, 512, 512]]
    setup['weights'] = style_weights + [0]
    Crosstyle_t = [CrossGramMatrix()(A, B).detach() for A, B in
                         zip(vgg(style_image, style_layers[:4]), vgg(style_image, style_layers[1:]))]
    content_t = [A.detach() for A in vgg(content_image, content_layers)]

    for iteration in range(setup['max_iter']):

        yolact.change_pred_outs(True)
        optimze_attack(optimizer, patch, content_mask, content_image, targets_image, masks_image, num_crowds, Crosstyle_t, content_t, setup)
        patch.data = normalize(torch.clamp(undo_normalize(patch.clone()), 0, 1))  # Keep the patch between 0 and 1

    return patch


if __name__ == '__main__':
    nb = int(sys.argv[1])
    nb2 = int(sys.argv[2])

    attack_setup = {
        'max_iter': 200,
        'lr': 0.02,
        'Texture_style': -1,
        'adv_loss_weight': 1,
        'textural_loss_weight': 200_000,
        'ssim_loss_weight': 50, #10,
        'tv_loss_weight': 0.00025,#0.00001,
        'yolact_weights': {'class_weight': 5, 'segmentation_weight': 10},
        'threshold_yolact': 0.5,
        'brightness': (0.8, 1.2),
        'contrast': (0.8, 1.2),
        'saturation': (0.8, 1.2),
        'hue': (-0.02, 0.02),
        'noise': (-0.02, 0.02),
        'transform': 0.2,
        'kernel_size': [1, 3, 5, 7, 9, 11],
        'sigma': (0.1, 3),
        'apply_color_manipulation': 0.5,
        'apply_transform': 0.5,
        'apply_gaussian': 0.25,
        'apply_jpeg': 0.75,
        'quality_factor': (2.25, 2.75),
        'weights': None
    }

    file_min_cost = open('/home/smg/v-marc/data/minimal_texture_cost.pkl', 'rb')
    print('For each attack we take the style image which produce the minimal texture loss.')
    print(file_min_cost)
    minimal_texture_cost_list = pickle.load(file_min_cost)
    file_min_cost.close()

    style_id = minimal_texture_cost_list
    path = 'result'  # Change the path for save
    print(f'save is here {path}')

    for i in tqdm(range(nb, nb2)):
        attack_setup['Texture_style'] = style_id[i]
        img_adv = fashionAdv_attack(imgs_to_attack[i], attack_setup)[0]

        img_adv_undo = undo_normalize(img_adv)
        save_image(img_adv_undo, f'/home/smg/v-marc/code/fashionAdv_results/{path}/patch_{dataset.ids[imgs_to_attack[i]]:012d}.png')

    print("ok")
