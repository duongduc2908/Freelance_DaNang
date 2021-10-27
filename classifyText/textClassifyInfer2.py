import cv2 
import numpy as np
import os 
from PIL import Image

# from mmocr1.getmodel import MMOCR

# from pan.predict import Pytorch_model

# from textClassify.product_classifier_infer import ClassifierInfer

from CRAFTpytorch.inference import load_model, extract_wordbox

from textRecognition.inferer import TextRecogInferer, default_args, text_recog

# from textRecognition.inferer import TextRecogInferer, default_args, text_recog
import sys
sys.path.append('classifyText')

# from mmocr1.getmodel import MMOCR
# from pan.predict import Pytorch_model

from newpan.infer import load_model_pan, extract_wordbox_pan

from textClassify.product_classifier_infer import ClassifierInfer, EnsembleInfer

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

# get 4 points of rectangle from poly
def get_box_from_poly(pts):
    pts = pts.reshape((-1,2)).astype(int)
    x,y,w,h = cv2.boundingRect(pts)
    area = w*h
    return np.array([x,y,x+w,y+h]), area

def ensemble(pts1, pts2):
    # take pts2 first then pts1
    iou_thres = 0.3
    pts = []
    
    # box2 = []
    # for poly in pts2:
    #     box2.append(get_box_from_poly(poly))
    
    # rm = []
    # for count, poly in enumerate(pts1):
    #     box = get_box_from_poly(poly)
    #     iou_score = np.array([bb_intersection_over_union(box, box2i) for box2i in box2])
    #     if np.any(iou_score) > iou_thres:
    #         rm.append(count)
    # pts1 = np.delete(pts1, rm, axis=0)
    
    if len(pts1) == 0:
        pts = pts2
    elif len(pts2) == 0:
        pts = pts1
    elif len(pts1)==0 and len(pts2)==0: pts = []
    else:
        pts = np.vstack((pts1, pts2))   
    
    rm = []
    for i in range(len(pts)):
        for count, poly in enumerate(pts[i+1:]):
            j = count +i+1
            boxi, areai = get_box_from_poly(pts[i])
            boxj, areaj = get_box_from_poly(poly)
            iou_score = bb_intersection_over_union(boxi, boxj)
            if iou_score > iou_thres:
                if areai > areaj: rm.append(j)
                else: rm.append(i)
                
    pts = np.delete(pts, rm, axis=0)
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

def textSpotting(config_file,detect1, detect2, recog, img, max_word=16):
    # mmocr_det_res = detect2.readtext(img=[img.copy()])
    # pts_mmocr = np.array([np.array(pts[:8]).reshape((-1,2)) for pts in mmocr_det_res[0]['boundary_result']])
    # pts_mmocr[np.where(pts_mmocr < 0)] = 0

    word_boxes = extract_wordbox(detect2, img)


    # preds, pts_pan, t = detect1.predict(img=img.copy())
    poly = extract_wordbox_pan(config_file, detect1, img)
    pts_pan = np.array([box.reshape((4,2)) for box in poly])
    pts_pan[np.where(pts_pan < 0)] = 0
    # pts_pan = expand_box(pts_pan)

    pts = ensemble(word_boxes, pts_pan) 
    # remove small text
    # pts = remove_small_text(pts.copy())
    # sort and take up to max_word poly
    pts = sort_pts(pts, max_word)
    yolo = convert_boxPoint2Yolo(pts.copy(), img.shape)
    
    crop_imgs = [crop_with_padding(img, poly) for poly in pts.copy()]
    for idx,crop in enumerate(crop_imgs):
        if crop is None:
            crop_imgs.pop(idx)
    # result_recog = recog.readtext(img=crop_imgs.copy(), batch_mode=True, single_batch_size=max_word)

    all_crop_img = [Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB)).convert('L') for item in crop_imgs]
    result_recog = recog_model.infer(images=all_crop_img)
    
    temp = {'boxPoint': None, 'boxYolo': None, 'text': None, 'text_score': None}
    result = []
    for count, poly in enumerate(pts):
        temp1 = temp.copy()
        temp1['boxPoint'] = poly
        temp1['boxYolo'] = yolo[count]
        temp1['text'] = result_recog[count][0]
        temp1['text_score'] = 1
        result.append(temp1)
        
    return result

def textClassify(model1, model2, model3, pre_result):
    text = ' '.join([res['text'] for res in pre_result])
    level1 = model1.product_predict([text])
    level2 = model2.product_predict([text])
    level3 = model3.product_predict([text])
    has_age = model1.check_has_age(text)
    
    pre_result.append((level1, level2, level3, has_age))
    return pre_result
    
def crop_with_padding(img, pts):
    pts_ = pts.reshape((-1, 2)).astype(int)

    rect = cv2.boundingRect(pts_)
    x,y,w,h = rect
    if w < 4 or h < 4:
        return None
    else:
        croped = img[y:y+h, x:x+w].copy()

        ## (2) make mask
        pts_ = pts_ - pts_.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts_], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg+ dst

        return dst2

def word2line(result, img):
    temp = {'center': None, 'text': None}
    new_res = []
    zero_mask = np.zeros(img.shape[:2]).astype('uint8')
    zero_mask_copy = zero_mask.copy()
    for res in result:
        x,y,w,h = cv2.boundingRect(res['boxes'].astype(int))
        zero_mask[y+int(0.2*h):y+int(0.8*h), x:x+w] = 125
        # zero_mask = cv2.polylines(zero_mask, [res['boxes'].astype(int)], True, 255, -1)

        center = np.array([x+0.5*w, y+0.5*h]).astype(int)
        # print(cv2.pointPolygonTest(res['boxes'].astype(int),tuple(center),False))
        item = temp.copy()
        item['center'] = center
        item['text'] = res['text']
        new_res.append(item)

    kernel = np.ones((1, 20), np.uint8)    
    zero_mask = cv2.dilate(zero_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # zero_mask_copy = cv2.drawContours(zero_mask_copy, contours, -1, 255, 2)
    # cv2.imrite('mask.jpg', zero_mask_copy)

    temp = {'contour': None, 'text': None, 'box': None}
    final_res = []  
    for contour in contours:
        box = cv2.boundingRect(contour.astype(int))
        item = temp.copy()
        item['box'] = np.array(box)
        item['contour'] = contour

        text_with_center = []
        temp1 = {'center': None, 'text': None}
        for pt in new_res:
            if cv2.pointPolygonTest(contour,tuple(pt['center']),False) > 0:
                item1 = temp1.copy()
                item1['text'] = pt['text']
                item1['center'] = pt['center']
                text_with_center.append(item1)
        
        text_with_center = np.array(text_with_center)
        only_center = [it['center'][0] for it in text_with_center]
        text_with_center = text_with_center[np.argsort(only_center)]
        
        item['text'] = ' '.join([text['text'] for text in text_with_center])
        final_res.append(item)

    return final_res

class default_args_v1():
    def __init__(self,model_path):
        self.workers = 0
        self.batch_size = 150
        self.saved_model = model_path
        self.batch_max_length = 80
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = """ầỏỉờbdúnuyỷủổrkxlqằwịộc,ẹựàề#đg;a)ĩv5iìừùâý6ễh0ọ/mỵẽ9ẵứ>ỗ"ã?ă8!ấ`ảỳf&ạ(3êửếz2ô4|ụểõ+1éởốồữậưũepỡó.ẩá“~ỹ=oệèơẻ:òs7íj”ợ%ặẳớ*ắ' tẫ-"""
        self.sensitive = False
        self.PAD = False
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256

# if __name__=='__main__':
    
#     # Load models detect into memory
#     detect_model = load_model('CRAFTpytorch/craft_mlt_25k.pth')

#     config_file = 'newpan/config.py'
#     model_path = 'newpan/checkpoint/pan.pth.tar'
        
#     pan_detect =  load_model_pan(config_file, model_path)
    
#     # load model recog
#     model_path = 'textRecognition/CRNN_finetune_v1.pth'
#     opt = default_args_v1(model_path)
#     recog_model = TextRecogInferer(opt)

#     # inference
#     img = cv2.imread('TextSpottingAndTextClassify/img_1.jpg')
    
#     result = textSpotting(config_file, pan_detect, detect_model, recog_model, img)

#     # img_save = cv2.polylines(img, [res['boxPoint'].astype(int) for res in result], True, (0,255,0), 1)
#     # cv2.imwrite('test.jpg', img_save)

#     ################## TextClassify ###################3333
#     classifyModel_level1 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level1_f192.15.pkl')
#     classifyModel_level2 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level2_f19306.pkl')
#     classifyModel_level3 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level3_f1_8117.pkl')

#     result = textClassify(classifyModel_level1, classifyModel_level2, classifyModel_level3, result)

#     ################## TextSpoting for banner's image###############

#     # load model recog
#     model_path = 'textRecognition/best_accuracy.pth'
#     opt = default_args(model_path)
#     inferer = TextRecogInferer(opt)
    
#     # detect text
#     # cho nay o truyen vao img va boxes
#     # img = cv2.polylines(img, np.array(boxes).astype(int), True, (255,255,255), -1)
    
#     word_boxs = extract_wordbox(detect_model, img)
#     # recog
#     # result = text_recog(img, word_boxs, inferer)

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


def spotting_text(pan_detect, craft_detect, mmocr_recog, img):
    config_file = 'classifyText/newpan/config.py'
    result = textSpotting(config_file, pan_detect, craft_detect, mmocr_recog, img)
    return result


def predict(text, classifyModel_level1, classifyModel_level3, branch=False, step=False, added_text=''):
    if branch:
        result = textClassify_branch(classifyModel_level1, text)
    if step:
        temp = {'boxPoint': None, 'boxYolo': None, 'text': None, 'text_score': None}
        temp['text'] = added_text
        text.append(temp)
        result = textClassify_step(classifyModel_level3, text)
    return result


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
        'abbott', 'bellamyorganic', 'blackmores', 'bubs_australia', 'danone', 'f99foods',
        'friesland_campina_dutch_lady',
        'gerber', 'glico', 'heinz', 'hipp', 'humana uc', 'mead_johnson', 'megmilksnowbrand', 'meiji', 'morigana',
        'namyang', 'nestle', 'no_brand', 'nutifood', 'nutricare', 'pigeon', 'royal_ausnz', 'vinamilk',
        'vitadairy', 'wakodo'
    }

    def __init__(self, n=3, ):
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

# def load_model(
#         mmocr_config_dir='classifyText/mmocr1/configs/',
#         pannet_model_path='classifyText/pan/pretrain/pannet_wordlevel.pth',
#         brand_text_model_path='classifyText/textClassify/checkpoints/product/product_classifier_level1.pkl',
#         step_model_dir_path="classifyText/textClassify/checkpoints/product/brands/",
# ):
#     mmocr_recog = MMOCR(det=None, recog='SAR', config_dir=mmocr_config_dir,)

#     model_path = pannet_model_path
#     pan_detect = Pytorch_model(model_path, gpu_id=0)

#     classifyModel_level1 = ClassifierInfer(path=brand_text_model_path, )
#     dict_model = {}
#     for key, link in mapping_checkpoints.items():
#         dict_model[key] = ClassifierInfer(path=step_model_dir_path + link)

#     return mmocr_recog, pan_detect, classifyModel_level1, dict_model

def load_model(
        recog_model_dir='classifyText/mmocr1/configs/',
        pannet_model_path='classifyText/newpan/checkpoint/pan.pth.tar',
        brand_text_model_path='classifyText/textClassify/checkpoints/product/product_classifier_level1.pkl',
        step_model_dir_path="classifyText/textClassify/checkpoints/product/brands/",
):
    # mmocr_recog = MMOCR(det=None, recog='SAR', config_dir=mmocr_config_dir,)

    # model_path = pannet_model_path
    # pan_detect = Pytorch_model(model_path, gpu_id=0)

    ## new model detect
    config_file = 'classifyText/newpan/config.py'
    model_path = pannet_model_path
        
    pan_detect =  load_model_pan(config_file, model_path)

    ## new model recog
    model_path = 'textSpotting/textRecognition/CRNN_finetune_v1.pth'
    opt = default_args_v1(model_path)
    recog_model = TextRecogInferer(opt)
    

    classifyModel_level1 = ClassifierInfer(path=brand_text_model_path, )
    dict_model = {}
    for key, link in mapping_checkpoints.items():
        dict_model[key] = ClassifierInfer(path=step_model_dir_path + link)

    return recog_model, pan_detect, classifyModel_level1, dict_model



