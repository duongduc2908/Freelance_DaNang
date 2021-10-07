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
from classifyText import textClassifyInfer
from classifyImage import objectClasssifyInfer
import hashlib

device = select_device(0)

def load_model_yolo(weights=['models/weights/binh_new_best.pt', 'models/weights/sua_new_best_2.pt'],
        ):
    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, graph_def = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    model_binh_sua = attempt_load(
        weights[0], map_location=device)  # load FP32 model
    model_sua = attempt_load(
        weights[1], map_location=device)  # load FP32 model
    return model_binh_sua.eval(),model_sua.eval()

craft_detect, model_recognition = textSpottingInfer.load_model_1()
mmocr_recog,pan_detect,classifyModel_level1,dict_model = textClassifyInfer.load_model()
chinh_model,model_step,labels_end,labels_branch,dict_middle,dict_step= objectClasssifyInfer.load_model()
model_binh_sua,model_sua = load_model_yolo()
print(list(model_binh_sua.parameters())[-1])

spell = SpellChecker(language=None,)  # loads default word frequency list
spell.word_frequency.load_text_file('corpus.txt')

with open('keywords.txt', 'r') as f:
    lines = f.readlines()
keywords = []
for line in lines:
    line = line.strip().split()
    keywords += line


def run(images):
    import numpy as np
    for image in images: print(image); print(image.shape); np.save('image1.npy', image)
    results_end = []
    stride_binh = int(model_binh_sua.stride.max())  # model stride
    names_binh = model_binh_sua.module.names if hasattr(model_binh_sua, 'module') else model_binh_sua.names  # get class names
    names_sua = model_sua.module.names if hasattr(model_sua, 'module') else model_sua.names  # get class names
    imgsz_binh = check_img_size(imgsz=640, s=stride_binh)  # check image size

    # Load datasets
    dataset = LoadImages(images, img_size=imgsz_binh, stride=stride_binh)
    # Run inference
    model_binh_sua(torch.zeros(1, 3, imgsz_binh, imgsz_binh).to(
        device).type_as(next(model_binh_sua.parameters())))  # run once
    model_sua(torch.zeros(1, 3, imgsz_binh, imgsz_binh).to(
        device).type_as(next(model_sua.parameters())))  # run once

    t0 = time.time()
    count=0
    for img, im0s in dataset:
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
        pred_binh = model_binh_sua(img, augment=False, visualize=False)[0]
        pred_sua = model_sua(img, augment=False, visualize=False)[0]

        # NMS
        pred_binh = non_max_suppression(pred_binh, 0.4, iou_thres=0.45, classes=None, agnostic=True, max_det=1000)
        pred_sua = non_max_suppression(pred_sua, conf_thres=0.3, iou_thres=0.45, classes=None, agnostic=True, max_det=1000)
        t2 = time_sync()

        # Process predictions binh sua
        for i, (det_binh, det_sua) in enumerate(zip(pred_binh, pred_sua)):  # detections per image
            print('#' * 5 + 'process predicts binh sua' + '#' * 5)
            print(i)
            print(det_binh)
            print(det_sua)
            print('#' * 10)
            s, im0 = '', im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy()
            item['height_width'] = [im0.shape[0],im0.shape[1]]
            if len(det_binh):
                det_binh[:, :4] = scale_coords(
                    img.shape[2:], det_binh[:, :4], im0.shape).round()

                # Write results
                item['binh_bu'] = []
                item['num_vu'] = []
                item['tre_em'] = []
                for *xyxy, conf, cls in reversed(det_binh):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # c = int(cls)
                    if int(cls) == 0:
                        item['binh_bu'].append(xywh)
                        # im0 = plot_one_box(xyxy, im0, label='binh_bu', color=colors(c, True), line_width=3)
                    if int(cls) == 1:
                        item['num_vu'].append(xywh)
                        # im0 = plot_one_box(xyxy, im0, label='num_vu', color=colors(c, True), line_width=3)
                    if int(cls) == 2:
                        item['tre_em'].append(xywh)
                        # im0 = plot_one_box(xyxy, im0, label='tre_em', color=colors(c, True), line_width=3)
                    
            else:
                item['binh_bu'] = None
                item['num_vu'] = None
                item['tre_em'] = None
            count_sua = 0
            if len(det_sua):
                # Rescale boxes from img_size to im0 size
                det_sua[:, :4] = scale_coords(img.shape[2:], det_sua[:, :4], im0.shape).round()

                # Write results
                item['sua'] = []
                count_sua =0
                for *xyxy, conf, cls in reversed(det_sua):
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
                    # End crop
                    # path_img_i = "res/test_{0}_{1}.jpg".format(count,count_sua)
                    # cv2.imwrite(path_img_i,crop)
                    # crop = cv2.imread("1.jpg")
                    # print(np.mean(crop))
                    path_txt_i = "res/test_{0}_{1}.txt".format(count,count_sua)
                    f_txt_i = open(path_txt_i,"a")
                    # To classification
                    height_crop,width_crop,_ =  crop.shape
                    name_merge=''
                    temp_step = ''
                    if height_crop > 65 and  width_crop > 50:
                        output_brand, sc1 = objectClasssifyInfer.predict(
                            chinh_model, crop.copy(), return_features=False)

                        result_text_spotting = textClassifyInfer.spotting_text(
                            pan_detect, craft_detect, mmocr_recog, crop)
                        # print(result_text_spotting)
                        result = textClassifyInfer.predict(result_text_spotting.copy(
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
                        f_txt_i.writelines("output_final_branch: "+output_final_branch+"\n")
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
                                result_2 = textClassifyInfer.predict(
                                    result_text_spotting, classifyModel_level1, classifyModel_level3, step=True, added_text=''.replace(' ', '_'))
                                temp_step = result_2[-1][0].replace(
                                    " ", "_")
                                brand_text = meger_label_branch(
                                    labels_end, 2, temp_step)
                        esem = Ensemble(
                            output_final_branch, output_merge, temp_step, dict_middle, dict_step, text_list)
                        label = esem.run()
                        f_txt_i.writelines("esem: "+label+"\n")
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
                    
                    f_txt_i.writelines(label+"\n")
                    f_txt_i.writelines("Chinh: "+str(name_merge)+"\n")
                    f_txt_i.writelines("Thanh: "+str(temp_step)+"\n")
                    f_txt_i.writelines("text:"+str(text_list) +"\n")
                
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
                # f_txt.writelines("====TEXT BANNER===\n")
                # f_txt.writelines(str(list_text))
                # Output text banner
            else:
                item['text_banner']=[]
                item['sua']={
                        'toa_do': None,
                        'label':None,
                        'text':None
                    }
            cv2.imwrite(path_img,im0)
            # f_txt.writelines("text_banner:"+str(list_text) +"\n")
            results_end.append(item)
    return results_end


lists_image=[]
import glob
for path in glob.glob("/u01/Intern/TEST/*"):
    lists_image.append(cv2.imread(path))
run(lists_image)




