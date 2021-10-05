import os
import numpy as np
import cv2
import sys

sys.path.append('textSpotting')

from CRAFTpytorch.inference import load_model, extract_wordbox

from textRecognition.inferer import TextRecogInferer, default_args, text_recog


def get_box_from_poly(pts):
    pts = pts.reshape((-1, 2)).astype(int)
    x, y, w, h = cv2.boundingRect(pts)
    return np.array([x, y, x + w, y + h])


def load_model_1(
        model_path_recognition='textSpotting/textRecognition/best_accuracy.pth',
        craft_path='textSpotting/CRAFTpytorch/craft_mlt_25k.pth',
):
    # Recognition
    opt = default_args(model_path_recognition)
    model_recognition = TextRecogInferer(opt)

    # Detection
    craft_detect = load_model(craft_path)
    return craft_detect, model_recognition


def predict(img, craft_detect, model_recognition):
    # detect text
    # cho nay o truyen vao img va boxes
    # img = cv2.polylines(img, np.array(boxes).astype(int), True, (255,255,255), -1)
    word_boxs = extract_wordbox(craft_detect, img)
    result = text_recog(img, word_boxs, model_recognition)
    return result
