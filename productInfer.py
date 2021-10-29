# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
from PIL import ImageFont, ImageDraw, Image
import argparse
import sys
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import glob
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.ensemble_labels import display_image,meger_label_branch,word2line,Ensemble,Analyzer
from spellchecker import SpellChecker

from utils.general import xywh2xyxy,clip_coords
from textSpotting import textSpottingInfer
from classifyText import textClassifyInfer2
from classifyImage import objectClasssifyInfer
import hashlib

device = select_device(0)

def load_model_yolo(weights=['models/weights/best.pt'],
        ):
    model_object_detect = attempt_load(
        weights[0], map_location=device)  # load FP32 model
    return model_object_detect.eval()

craft_detect, model_recognition = textSpottingInfer.load_model_1()
mmocr_recog,pan_detect,classifyModel_level1,dict_model = textClassifyInfer2.load_model()
chinh_model,model_step,labels_end,labels_branch,dict_middle,dict_step= objectClasssifyInfer.load_model()
model_object_detect = load_model_yolo()

spell = SpellChecker(language=None,)  # loads default word frequency list
spell.word_frequency.load_text_file('corpus.txt')

with open('keywords.txt', 'r') as f:
    lines = f.readlines()
keywords = []
for line in lines:
    line = line.strip().split()
    keywords += line


def run(images):
    results_end = []
    stride_object_detect = int(model_object_detect.stride.max())  # model stride
    imgsz_object_detect = check_img_size(imgsz=640, s=stride_object_detect)  # check image size

    # Load datasets
    dataset = LoadImages(images, img_size=imgsz_object_detect, stride=stride_object_detect)
    # Run inference
    model_object_detect(torch.zeros(1, 3, imgsz_object_detect, imgsz_object_detect).to(
        device).type_as(next(model_object_detect.parameters())))  # run once

    count=0
    for img, im0s in dataset:
        os.makedirs("res",exist_ok=True)
        count+=1
        path_img = "res/test_{0}.jpg".format(count)
        item = {}
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        preds = model_object_detect(img, augment=False, visualize=False)[0]

        # NMS
        preds = non_max_suppression(preds, 0.4, iou_thres=0.45, classes=None, agnostic=True, max_det=1000)

        # Process predictions binh sua
        for i, pred in enumerate(preds):  # detections per image
            s, im0 = '', im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy()
            item['height_width'] = [im0.shape[0],im0.shape[1]]
            if len(pred):
                pred[:, :4] = scale_coords(
                    img.shape[2:], pred[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(pred):
                    # Write results
                    item['binh_bu'] = []
                    item['num_vu'] = []
                    item['tre_em'] = []
                    item['sua'] = []
                    count_sua =0
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    if int(cls) in [2,3,4]:
                        if int(cls) == 2:
                            item['binh_bu'].append(xywh)
                            im0 = plot_one_box(xyxy, im0, label='binh_bu', color=colors(c, True), line_width=3)
                        if int(cls) == 3:
                            item['num_vu'].append(xywh)
                            im0 = plot_one_box(xyxy, im0, label='num_vu', color=colors(c, True), line_width=3)
                        if int(cls) == 4:
                            item['tre_em'].append(xywh)
                            im0 = plot_one_box(xyxy, im0, label='tre_em', color=colors(c, True), line_width=3)
                    else:
                        item['binh_bu'] = None
                        item['num_vu'] = None
                        item['tre_em'] = None

                    if int(cls) in [0,1]:
                        count_sua+=1
                        text_list = [] 
                        # Output hop sua
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # Xu ly box 
                        BGR = True
                        xyxy_test = torch.tensor(xyxy).view(-1, 4)
                        b = xyxy2xywh(xyxy_test)  # boxes
                        b[:, 2:] = b[:, 2:] * 1.02 + \
                            10  # box wh * gain + pad
                        xyxy_crop = xywh2xyxy(b).long()
                        clip_coords(xyxy_crop, imc.shape)
                        crop = imc[int(xyxy_crop[0, 1]):int(xyxy_crop[0, 3]), int(
                            xyxy_crop[0, 0]):int(xyxy_crop[0, 2]), ::(1 if BGR else -1)]
                        
                        ignord = np.zeros_like(crop)
                        imc[int(xyxy_crop[0, 1]):int(xyxy_crop[0, 3]), int(xyxy_crop[0, 0]):int(xyxy_crop[0, 2]), ::(1 if BGR else -1)] = ignord
                        
                        # To classification
                        height_crop,width_crop,_ =  crop.shape
                        name_merge=''
                        temp_step = ''
                        if height_crop > 65 and  width_crop > 50:
                            output_brand, sc1 = objectClasssifyInfer.predict(
                                chinh_model, crop.copy(), return_features=False)

                            result_text_spotting = textClassifyInfer2.spotting_text(
                                pan_detect, craft_detect, mmocr_recog, crop)
                            # print(result_text_spotting)
                            result = textClassifyInfer2.predict(result_text_spotting.copy(
                            ), classifyModel_level1, classifyModel_level3=None, branch=True)

                            branch_0 = result[-1][0][0].replace(" ", "_")
                            for i in result_text_spotting[:-1]:
                                text = i['text'].lower().replace(' ', '_')
                                text_list.append(text)
                            test_keyword = False
                            for text_ in text_list:
                                if text_ in keywords:
                                    test_keyword = True
                                    break
                            c = (list(labels_branch.keys())
                                [output_brand].strip())
                            if test_keyword == True:
                                output_final_branch = result[-1][0][0]
                            elif len(text_list) == 0 and sc1 < 0.98:
                                output_final_branch = 'Unknow'
                            elif len(text_list) >= 4:
                                branch_0 = c
                                if sc1 > 0.95:
                                    output_final_branch = c
                                else:
                                    output_final_branch = 'Unknow'
                            else:
                                if sc1 > 0.93:
                                    output_final_branch = c
                                else:
                                    output_final_branch = 'Unknow'
                            output_final_branch = output_final_branch.replace(" ", "_")
                            label = output_final_branch
                            check_list = False
                            output_merge, _ = objectClasssifyInfer.predict_merge_model(
                                model_step, crop)
                            if len(dict_middle[str(output_merge)]) > 1:
                                check_list = True
                            name_merge = dict_middle[str(
                                output_merge)][-1]
                            brand_merge = name_merge.split("/")[0]

                            temp_step = None
                            if output_final_branch in ["f99foods", "heinz", "bubs_australia", "megmilksnowbrand", "meiji"]:
                                pass
                            else:
                                if output_final_branch in dict_model.keys():
                                    classifyModel_level3 = dict_model[output_final_branch]                                          
                                    result_2 = textClassifyInfer2.predict(
                                        result_text_spotting, classifyModel_level1, classifyModel_level3, step=True, added_text=''.replace(' ', '_'))
                                    temp_step = result_2[-1][0].replace(
                                        " ", "_")
                                    brand_text = meger_label_branch(
                                        labels_end, 2, temp_step)
                            esem = Ensemble(
                                output_final_branch, output_merge, temp_step, dict_middle, dict_step, text_list)
                            label = esem.run()
                            if width_crop / height_crop < 0.495: #(4/7):
                                if "yoko" in label.split("/"):
                                    pass
                                else:
                                    label = output_final_branch
                                    if label == "f99foods":
                                        label = 'f99//'
                                
                        else:
                            label = "size nho ({0} x {1})".format(width_crop,height_crop)
                        c = int(cls)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=3)
                        # Output label
                        item['sua'].append({
                            'toa_do': xywh,
                            'label':label,
                            'text':text_list
                        })
                
                # Text banner
                result_text = textSpottingInfer.predict(
                    imc, craft_detect, model_recognition)
                final_res = word2line(result_text, imc)
                list_text = []
                for res in final_res:
                    x1, y1, w, h = res['box']
                    x2 = x1+w
                    y2 = y1+h
                    text = res['text']
                    list_text.append((text))

                item['text_banner']= list_text
            else:
                item['text_banner']=[]
                item['sua']={
                        'toa_do': None,
                        'label':None,
                        'text':None
                    }
            cv2.imwrite(path_img,im0)
            results_end.append(item)
    return results_end


if __name__ =='__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--dir', type=str, default='', help='specific what task you want to excute')
    opt = arg.parse_args()
    lists_image=[]
    for path in glob.glob(opt.dir + "/*"):
        lists_image.append(cv2.imread(path))
    run(lists_image)
