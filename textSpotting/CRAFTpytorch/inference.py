import os 
import torch 
import numpy as np
from collections import OrderedDict
import cv2 
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from .craft import CRAFT
from . import imgproc
from . import craft_utils

def crop_text(img, save_name, pts):
    pts_ = pts.reshape(-1, 2).astype(int)

    rect = cv2.boundingRect(pts_)
    x,y,w,h = rect
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

    cv2.imwrite(save_name, dst2)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def extract_wordbox(net, image):
    text_threshold = 0.7
    link_threshold = 0.4
    low_text = 0.4
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    if boxes != []:
        boxes[np.where(boxes < 0)] = 0
        boxes = expand_box(boxes)
        
    return boxes

def expand_box(boxes):
    new_boxes = []
    for box in boxes:
        rrect = cv2.minAreaRect(box)
        temp = (rrect[1][0], rrect[1][1])
        if temp[0]<temp[1]: temp = (temp[0]*1.2, temp[1])
        else: temp = (temp[0], temp[1]*1.2)
        new_rrect = (rrect[0], temp, rrect[2])
        new_box = cv2.boxPoints(new_rrect)
        new_box[np.where(new_box<0)] = 0
        new_boxes.append(new_box)
        
    return new_boxes

def load_model(model_path):
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(model_path)))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    net.eval()
    return net

if __name__ == '__main__':
    #load model
    model = load_model('craft_mlt_25k.pth')
    #inference
    img = cv2.imread('img_0.jpg')
    boxes = extract_wordbox(model, img)
    
    #save crop text
    save_folder = 'result'
    for count, poly in enumerate(boxes):
        x,y,w,h = cv2.boundingRect(poly)
        if h > 10 and w > 10:
            # crop = img.copy()[y:y+h, x:x+w, :]
            # cv2.imwrite(os.path.join('res_img', img_file[:-4]+'_'+str(count)+'.jpg'), crop)
            save_name = os.path.join(save_folder, str(count)+'.jpg')
            crop_text(img, save_name, poly)