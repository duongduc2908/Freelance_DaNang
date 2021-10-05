import re
import cv2 
import numpy as np
import os 
import ngram
import sys
sys.path.append('classifyText' )

from mmocr1.getmodel import MMOCR
from pan.predict import Pytorch_model
from textClassify.product_classifier_infer import ClassifierInfer,EnsembleInfer

mapping_checkpoints = {
    "abbott": "product_classifier_level3_abbott.pkl",
    "bellamyorganic": "product_classifier_level3_bellamyorganic.pkl",
    "blackmores": "product_classifier_level3_blackmores.pkl",
    "danone": "product_classifier_level3_danone.pkl",
    "friesland_campina_dutch_lady": "product_classifier_level3_friesland campina dutch lady.pkl",
    "gerber": "product_classifier_level3_gerber.pkl",
    "glico": "product_classifier_level3_glico.pkl",
    "hipp": "product_classifier_level3_hipp.pkl",
    "humana_uc": "product_classifier_level3_humana uc.pkl",
    "mead_johnson": "product_classifier_level3_mead johnson.pkl",
    "meiji": "product_classifier_level3_meiji.pkl",
    "morigana": "product_classifier_level3_morigana.pkl",
    "namyang": "product_classifier_level3_namyang.pkl",
    "nestle": "product_classifier_level3_nestle.pkl",
    "nutifood": "product_classifier_level3_nutifood.pkl",
    "nutricare": "product_classifier_level3_nutricare.pkl",
    "pigeon": "product_classifier_level3_pigeon.pkl",
    "royal_ausnz": "product_classifier_level3_royal ausnz.pkl",
    "vinamilk": "product_classifier_level3_vinamilk.pkl",
    "vitadairy": "product_classifier_level3_vitadairy.pkl",
    "wakodo": "product_classifier_level3_wakodo.pkl"
}

class Analyzer:
    brands = {
        'abbott', 'bellamyorganic', 'blackmores', 'bubs_australia', 'danone', 'f99foods', 'friesland_campina_dutch_lady',
        'gerber', 'glico', 'heinz', 'hipp', 'humana uc', 'mead_johnson', 'megmilksnowbrand', 'meiji', 'morigana',
        'namyang', 'nestle', 'no_brand', 'nutifood', 'nutricare', 'pigeon', 'royal_ausnz', 'vinamilk',
        'vitadairy', 'wakodo'
    }

    def __init__(self, n=3,):
        self.n = n
        self.index = ngram.NGram(N=n)

    def __call__(self, s):
        tokens = re.split(r'\s+', s.lower().strip())
        filtered_tokens = []
        for token in tokens:
            if len(token) > 20:
                continue

            if re.search(r'[?\[\]\(\):!]', token):
                continue

            if re.search(f'\d{2,}', token):
                continue

            filtered_tokens.append(token)

        non_ngram_tokens = []
        ngram_tokens = []

        for token in filtered_tokens:
            if token in self.brands:
                non_ngram_tokens.append(token)
                n_grams = list(self.index.ngrams(self.index.pad(token)))
                ngram_tokens.extend(n_grams)
            else:
                n_grams = list(self.index.ngrams(self.index.pad(token)))
                ngram_tokens.extend(n_grams)
        res = [*non_ngram_tokens, *ngram_tokens]
        return res

sys.path.append('textSpotting/CRAFTpytorch' )
from CRAFTpytorch.inference import extract_wordbox

def load_model():
    mmocr_recog = MMOCR(det=None, recog='SAR', config_dir='classifyText/mmocr1/configs/')

    model_path = 'classifyText/pan/pretrain/pannet_wordlevel.pth'
    pan_detect = Pytorch_model(model_path, gpu_id=0)

    classifyModel_level1 = ClassifierInfer(path='classifyText/textClassify/checkpoints/product/product_classifier_level1.pkl')
    dict_model={}
    for key, link in mapping_checkpoints.items():
        dict_model[key] = ClassifierInfer(path="classifyText/textClassify/checkpoints/product/brands/"+link)
        
    return mmocr_recog,pan_detect,classifyModel_level1,dict_model

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_box_from_poly(pts):
    pts = pts.reshape((-1,2)).astype(int)
    x,y,w,h = cv2.boundingRect(pts)
    return np.array([x,y,x+w,y+h])

def ensemble(pts1, pts2):
    # take pts2 first then pts1
    iou_thres = 0.5
    pts = []
    box2 = []
    for poly in pts2:
        box2.append(get_box_from_poly(poly))
    
    rm = []
    for count, poly in enumerate(pts1):
        box = get_box_from_poly(poly)
        iou_score = np.array([bb_intersection_over_union(box, box2i) for box2i in box2])
        if np.any(iou_score) > iou_thres:
            rm.append(count)
    pts1 = np.delete(pts1, rm, axis=0)
    
    if pts1.size == 0:
        pts = pts2
    elif pts2.size == 0:
        pts = pts1
    elif pts1.size==0 and pts2.size==0: pts = []
    else:
        pts = np.vstack((pts1, pts2))   
    return pts

def convert_boxPoint2Yolo(pts, imgShape):
    yolo = []
    height = imgShape[0]
    width = imgShape[1]
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        x = (x+w/2)/width
        y = (y+h/2)/height
        yolo.append(np.array([x,y,w/width, h/height]))
    return yolo

def remove_small_text(pts):
    new_pts = []
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        if h > 10 and w > 10:
            new_pts.append(poly)
    return np.array(new_pts)

def sort_pts(pts, max_pts):
    heights = []    # sort by height
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        heights.append(h)
    new_poly = pts[np.argsort(heights)][::-1]
    return new_poly[:max_pts]

def expand_number2(boxes):
    new_boxes = []
    for box in boxes:
        rrect = cv2.minAreaRect(box.astype('float32'))
        temp = (rrect[1][0], rrect[1][1])
        if temp[0]>temp[1]: temp = (temp[0]*1.2, temp[1])
        else: temp = (temp[0], temp[1]*1.2)
        new_rrect = (rrect[0], temp, rrect[2])
        new_box = cv2.boxPoints(new_rrect)
        new_box[np.where(new_box<0)] = 0
        new_boxes.append(new_box)
        
    return new_boxes
def textSpotting(detect1, detect2, recog, img, max_word=16):
    word_boxes = extract_wordbox(detect2, img)
    preds, pts_pan, t = detect1.predict(img=img.copy())
    pts_pan[np.where(pts_pan < 0)] = 0

    pts = ensemble(word_boxes, pts_pan) 
    # sort and take up to max_word poly
    pts = sort_pts(pts, max_word)
    yolo = convert_boxPoint2Yolo(pts, img.shape)
    
    crop_imgs = [crop_with_padding(img, poly) for poly in pts]
    
    result_recog = recog.readtext(img=crop_imgs.copy(), batch_mode=True, single_batch_size=max_word)
    
    temp = {'boxPoint': None, 'boxYolo': None, 'text': None, 'text_score': None}
    result = []
    for count, poly in enumerate(pts):
        temp1 = temp.copy()
        temp1['boxPoint'] = poly
        temp1['boxYolo'] = yolo[count]
        temp1['text'] = result_recog[count]['text']
        temp1['text_score'] = result_recog[count]['score']
        result.append(temp1)
     ######### refine text value of number 3, expand box for case of 2 with 20%
    for count, res in enumerate(result):
        if res['text'] == '2':
            new_pts = [res['boxPoint']]
            new_pts = expand_number2(new_pts)
            result[count]['boxPoint'] = new_pts[0]

            new_img = [crop_with_padding(img, poly) for poly in new_pts]
            new_res = recog.readtext(img=new_img.copy(), batch_mode=False)
            result[count]['text'] = new_res[0]['text']
        
    return result

def textClassify_branch(model1, pre_result):
    text = ' '.join([res['text'] for res in pre_result])
    level1 = model1.product_predict_branch([text])
    has_age = model1.check_has_age(text)
    pre_result.append((level1, has_age))
    return pre_result
                                    
def textClassify_step(model3, pre_result):
    text = ' '.join([res['text'] for res in pre_result])
    level3 = model3.product_predict([text])
    pre_result.append((level3))
    return pre_result
    
def crop_with_padding(img, pts):
    pts_ = pts.reshape((-1, 2)).astype(int)

    rect = cv2.boundingRect(pts_)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    pts_ = pts_ - pts_.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts_], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
    return dst2

def spotting_text(pan_detect, craft_detect, mmocr_recog, img):
    result = textSpotting(pan_detect, craft_detect, mmocr_recog, img)
    return result

def predict(text,classifyModel_level1,classifyModel_level3,branch=False,step=False,added_text =''):
    if branch:
        result = textClassify_branch(classifyModel_level1, text)
    if step:
        temp = {'boxPoint': None, 'boxYolo': None, 'text': None, 'text_score': None}
        temp['text'] = added_text
        text.append(temp)
        result = textClassify_step(classifyModel_level3, text)
    return result
