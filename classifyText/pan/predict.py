# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import torch
from torchvision import transforms
import os
import cv2
import time

from .models import get_model

from .post_processing import decode

import argparse

def main():
    parse = argparse.ArgumentParser(description="PAN Text detection")
    parse.add_argument('--trained_model', default='./pretrain/pan-resnet18-ic15.pth', type=str, help='pretrained model')
    parse.add_argument('--test_folder', default='./test_img', type=str, help='folder path to input images')
    parse.add_argument('--result_folder', default='./result', type=str, help='folder path to result images')

    args = parse.parse_args()
    
    # import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox

    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')

    model_path = args.trained_model
    img_path = args.test_folder
    result_path = args.result_folder
    
    model = Pytorch_model(model_path, gpu_id=0)
    list_img = []
    list_name = os.listdir(img_path)
    for path in list_name:
        list_img.append(os.path.join(img_path, path))
    
    for idx, path in enumerate(list_img):
        preds, boxes_list, t = model.predict(path)
        print('predict %dth image in %.2fs'%(idx, t))
        
        # cv2.imwrite(os.path.join(result_path, 'pred_'+list_name[idx]), preds)
        
        # img = draw_bbox(cv2.imread(path)[:, :, ::-1], boxes_list)
        img = draw_bbox(cv2.imread(path), boxes_list)
        cv2.imwrite(os.path.join(result_path, 'res_'+list_name[idx]), img)
    

    

    # # 初始化网络
    # model = Pytorch_model(model_path, gpu_id=0)
    # preds, boxes_list, t = model.predict(img_path)
    # show_img(preds)
    # img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
    # show_img(img, color=True)
    # plt.show()

    

def decode_clip(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    import numpy as np
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area:
            continue
        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)

        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(d_i))
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)


class Pytorch_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        self.net = get_model(config)

        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img, short_size: int = 736):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        # assert os.path.exists(img), 'file is not exists'
        # img = cv2.imread(img)
        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds, boxes_list = decode(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start
        return preds, boxes_list, t


if __name__ == '__main__':
    main()