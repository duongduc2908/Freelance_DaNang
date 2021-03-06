import base64
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
from utils.ensemble_labels import display_image, meger_label_branch, word2line, Ensemble, Analyzer
import numpy as np

from utils.general import xywh2xyxy, clip_coords
from textSpotting import textSpottingInfer
from classifyText import textClassifyInfer
from classifyImage import objectClasssifyInfer

model_path_recognition = 'textSpotting/textRecognition/best_accuracy.pth'
craft_path = 'textSpotting/CRAFTpytorch/craft_mlt_25k.pth'

mmocr_config_dir = 'classifyText/mmocr1/configs/'
pannet_model_path = 'classifyText/pan/pretrain/pannet_wordlevel.pth'
brand_text_model_path = 'classifyText/textClassify/checkpoints/product/product_classifier_level1.pkl'
step_model_dir_path = "classifyText/textClassify/checkpoints/product/brands/"

brand_image_classifier_model_path = "classifyImage/weights/model_final.h5"
step_image_classifier_model_path = "classifyImage/weights/model_step.h5"
labels_txt_fn = 'classifyImage/labels.txt'
labels_branchs_dict_fn = 'classifyImage/labels_branchs_dict.json'

yolo_model_paths = ['models/weights/binh_new_best.pt', 'models/weights/sua_new_best_2.pt']

keywords_fn = 'keywords.txt'
correct_corpus_fn = 'corpus.txt'


def load_keywords(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    keywords = []
    for line in lines:
        line = line.strip().split()
        keywords += line
    return keywords


def load_model_yolo(
        weights,
        device,
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
    return model_binh_sua.eval(), model_sua.eval()


def load_model(device):
    craft_detect, model_recognition = textSpottingInfer.load_model_1(
        model_path_recognition=model_path_recognition,
        craft_path=craft_path,
    )
    mmocr_recog, pan_detect, classifyModel_level1, dict_model = textClassifyInfer.load_model(
        mmocr_config_dir=mmocr_config_dir,
        pannet_model_path=pannet_model_path,
        brand_text_model_path=brand_text_model_path,
        step_model_dir_path=step_model_dir_path,
    )
    (
        chinh_model,
        model_step,
        labels_end,
        labels_branch,
        dict_middle,
        dict_step
    ) = objectClasssifyInfer.load_model(
        weight_path_1=brand_image_classifier_model_path,
        weight_path_2=step_image_classifier_model_path,
        labels_txt_fn=labels_txt_fn,
        labels_branchs_dict_fn=labels_branchs_dict_fn,
    )

    model_binh_sua, model_sua = load_model_yolo(
        weights=yolo_model_paths,
        device=device,
    )

    return (
        (craft_detect, model_recognition),
        (mmocr_recog, pan_detect, classifyModel_level1, dict_model),
        (
            chinh_model,
            model_step,
            labels_end,
            labels_branch,
            dict_middle,
            dict_step,
        ),
        (model_binh_sua, model_sua,),
    )


from PIL import Image
import io


class Infer:
    def __init__(
            self,
            craft_detect, model_recognition,

            mmocr_recog, pan_detect, classifyModel_level1, dict_model,

            chinh_model,
            model_step,
            labels_end,
            labels_branch,
            dict_middle,
            dict_step,

            model_binh_sua, model_sua,
            keywords, spell,
            device,
    ):
        for k, v in locals().items():
            setattr(self, k, v)
            
    def run(self, images):
        craft_detect, model_recognition = self.craft_detect, self.model_recognition

        (
            mmocr_recog, pan_detect, classifyModel_level1, dict_model
        ) = self.mmocr_recog, self.pan_detect, self.classifyModel_level1, self.dict_model,

        (
            chinh_model,
            model_step,
            labels_end,
            labels_branch,
            dict_middle,
            dict_step,
        ) = (
            self.chinh_model,
            self.model_step,
            self.labels_end,
            self.labels_branch,
            self.dict_middle,
            self.dict_step,
        )

        model_binh_sua, model_sua = (self.model_binh_sua, self.model_sua)
        device = self.device
        keywords = self.keywords
        spell = self.spell

        # return results_end
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
        for img, im0s in dataset:
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
                        if int(cls) == 1:
                            item['num_vu'].append(xywh)
                        if int(cls) == 2:
                            item['tre_em'].append(xywh)
                        
                else:
                    item['binh_bu'] = None
                    item['num_vu'] = None
                    item['tre_em'] = None
                if len(det_sua):
                    # Rescale boxes from img_size to im0 size
                    det_sua[:, :4] = scale_coords(img.shape[2:], det_sua[:, :4], im0.shape).round()

                    # Write results
                    item['sua'] = []
                    for *xyxy, conf, cls in reversed(det_sua):
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
                        # To classification
                        height_crop,width_crop,_ =  crop.shape
                        name_merge=''
                        temp_step = ''
                        if height_crop > 65 and  width_crop > 50:
                            output_brand, sc1 = objectClasssifyInfer.predict(
                                chinh_model, crop, return_features=False)
                            result_text_spotting = textClassifyInfer.spotting_text(
                                pan_detect, craft_detect, mmocr_recog, crop)

                            result = textClassifyInfer.predict(result_text_spotting.copy(
                            ), classifyModel_level1, classifyModel_level3=None, branch=True)
                            
                            branch_0 = result[-1][0][0].replace(" ", "_")
                            for i in result[:-1]:
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
                                    result_2 = textClassifyInfer.predict(
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
                    # Output text banner
                else:
                    item['text_banner']=[]
                    item['sua']={
                            'toa_do': None,
                            'label':None,
                            'text':None
                        }
                results_end.append(item)
        return results_end
