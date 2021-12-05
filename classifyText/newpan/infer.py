import torch
import numpy as np
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config
from PIL import Image
import torchvision.transforms as transforms

from tqdm import tqdm
import cv2

# from dataset import build_data_loader
# from models import build_model
from .models import PAN
from .models.utils import fuse_module
# from .utils import ResultFormat, AverageMeter, Corrector


def test(test_loader, model, cfg):
    model.eval()

    for idx, data in enumerate(tqdm(test_loader)):
        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))

        # forward
        with torch.no_grad():
            outputs = model(**data)


        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)


def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img



def load_model_pan(config_file, model_path):
    cfg = Config.fromfile(config_file)
    # model
    param = dict()
    for key in cfg.model:
        if key == 'type':
            continue
        param[key] = cfg.model[key]
        
    model = PAN(**param)
    # model = build_model(cfg.model)
    model = model.cuda()
    
    if model_path is not None:
        if os.path.isfile(model_path):
            print("Loading model and optimizer from checkpoint '{}'".format(model_path))
            sys.stdout.flush()

            checkpoint = torch.load(model_path)
            # print(checkpoint.keys())

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
            
            # state = dict(state_dict=model.state_dict)
            # torch.save(model.state_dict(), 'checkpoints/pan_r18_alldata2.pth.tar')
            
        else:
            print("No checkpoint found at '{}'".format(model_path))
            raise
        
    # fuse conv and bn
    # turn off batch norm layer to speed up model while inferencing
    model = fuse_module(model)
    model.eval()
    return model

def get_input(config_file, img):
    cfg = Config.fromfile(config_file)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=False
        ))
    
    # img = cv2.imread(img_path)
    img = img[:, :, [2,1,0]]
    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )

    short_size = 736
    img = scale_aligned_short(img, short_size)
    img_meta.update(dict(
        img_size=np.array(img.shape[:2])
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze_(0)

    data = dict(
        imgs=img.cuda(),
        img_metas=img_meta
    )
    data.update(dict(cfg=cfg))
    return data

def extract_wordbox_pan(config_file, model, img):
    model_input = get_input(config_file, img)
    with torch.no_grad():
        outputs = model(**model_input)
    poly = outputs['bboxes']
    return poly

if __name__ == '__main__':
        
    # parser = argparse.ArgumentParser(description='Hyperparams')
    # parser.add_argument('config', type=str, nargs='?',help='config file path', default='config/pan/pan_r18_ic15.py')
    # parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_alldata2/checkpoint.pth.tar')
    # parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_alldata2.pth.tar')
    # parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_synth.pth.tar')
    # args = parser.parse_args()

    # print(args.checkpoint)
    # print(args.config)
    
    config_file = 'config.py'
    model_path = 'checkpoint/pan.pth.tar'
        
    model =  load_model(config_file, model_path)
    
    img = cv2.imread('../TextSpottingAndTextClassify/img_1.jpg')
    poly = extract_wordboxes(config_file, model, img)
    
    img_save = cv2.polylines(img, [box.reshape((4,2)) for box in poly], True, (0,255,0), 1)
    cv2.imwrite('test1.jpg', img_save)
    
    # cfg = Config.fromfile(args.config)
    # print(cfg['model'])
