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
from classifyText import textClassifyInfer2
from classifyImage import objectClasssifyInfer

model_path_recognition = 'textSpotting/textRecognition/best_accuracy.pth'
craft_path = 'textSpotting/CRAFTpytorch/craft_mlt_25k.pth'

recog_model_dir = 'classifyText/mmocr1/configs/'
pannet_model_path = 'classifyText/newpan/checkpoint/pan.pth.tar'
brand_text_model_path = 'classifyText/textClassify/checkpoints/product/product_classifier_level1.pkl'
step_model_dir_path = "classifyText/textClassify/checkpoints/product/brands/"

brand_image_classifier_model_path = "classifyImage/weights/model_final.h5"
step_image_classifier_model_path = "classifyImage/weights/model_step.h5"
labels_txt_fn = 'classifyImage/labels.txt'
labels_branchs_dict_fn = 'classifyImage/labels_branchs_dict.json'

yolo_model_paths = ['models/weights/best.pt']

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
    model_object_detect = attempt_load(
        weights[0], map_location=device)  # load FP32 model
    return model_object_detect.eval()


def load_model(device):
    craft_detect, model_recognition = textSpottingInfer.load_model_1(
        model_path_recognition=model_path_recognition,
        craft_path=craft_path,
    )
    mmocr_recog, pan_detect, classifyModel_level1, dict_model = textClassifyInfer2.load_model(
        recog_model_dir=recog_model_dir,
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

    model_object_detect= load_model_yolo(
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
        model_object_detect,
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

            model_object_detect,
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

        model_object_detect = (self.model_object_detect)
        device = self.device
        keywords = self.keywords
        spell = self.spell

        # return results_end
        results_end = []
        stride_object_detect = int(model_object_detect.stride.max())  # model stride
        imgsz_object_detect = check_img_size(imgsz=640, s=stride_object_detect)  # check image size

        # Load datasets
        dataset = LoadImages(images, img_size=imgsz_object_detect, stride=stride_object_detect)
        # Run inference
        model_object_detect(torch.zeros(1, 3, imgsz_object_detect, imgsz_object_detect).to(
            device).type_as(next(model_object_detect.parameters())))  # run once
        for img, im0s in dataset:
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
            for i, pred in enumerate(preds):
                s, im0 = '', im0s.copy()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                imc = im0.copy()
                item['height_width'] = [im0.shape[0],im0.shape[1]]
                item['binh_bu'] = []
                item['num_vu'] = []
                item['tre_em'] = []
                item['sua'] = []
                if len(pred):
                    pred[:, :4] = scale_coords(
                        img.shape[2:], pred[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(pred):
                        # Write results
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if int(cls) in [2,3,4]:
                            if int(cls) == 2:
                                item['binh_bu'].append(xywh)
                            if int(cls) == 3:
                                item['num_vu'].append(xywh)
                            if int(cls) == 4:
                                item['tre_em'].append(xywh)

                        if int(cls) in [0,1]:
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
                            # To classification
                            height_crop,width_crop,_ =  crop.shape
                            name_merge=''
                            temp_step = ''
                            if height_crop > 65 and  width_crop > 50:
                                output_brand, sc1 = objectClasssifyInfer.predict(
                                    chinh_model, crop.copy(), return_features=False)

                                result_text_spotting = textClassifyInfer2.spotting_text(
                                    pan_detect, craft_detect, mmocr_recog, crop)
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
                results_end.append(item)
        return results_end
