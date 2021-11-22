import tensorflow as tf
import torch
import numpy as np
import cv2
import json
import sys

sys.path.append('classifyImage')
from source.models.mobilenetv2 import MobileNetV2


with open("classifyImage/labels_step.json", "r") as f:
    dict_step = json.load(f)
with open("classifyImage/labels_middle.json", "r") as f1:
    dict_middle = json.load(f1)

# tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 3)])
        logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)


def load_model(
        weight_path_1="classifyImage/weights/model_final.h5",
        weight_path_2="classifyImage/weights/model_step.pt",
        labels_txt_fn='classifyImage/labels.txt',
        labels_branchs_dict_fn='classifyImage/labels_branchs_dict.json',device='cuda:0'
):
    with open(labels_txt_fn, 'r') as f:
        labels = f.readlines()
    labels_end = [x.strip() for x in labels]
    with open(labels_branchs_dict_fn, 'r') as f:
        labels_branch = json.load(f)

    model_step = MobileNetV2(76)
    model_step.load_state_dict(torch.load(weight_path_2)['state_dict'])
    model_step = model_step.to(device)
    model_step.eval()

    return (
        tf.keras.models.load_model(weight_path_1, compile=False),
        model_step,
        labels_end,
        labels_branch,
        dict_middle, dict_step
    )


def predict(model, img, return_features=True):
    height, width, _ = img.shape
    if height > width:
        img = np.pad(
            img,
            [[0, 0], [(height - width) // 2, (height - width) // 2], [0, 0]],
            mode='constant',
            constant_values=255,
        )
    else:
        img = np.pad(
            img,
            [[(width - height) // 2, (width - height) // 2], [0, 0], [0, 0]],
            mode='constant',
            constant_values=255,
        )
    img = cv2.resize(img, (96, 96))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float') / 255.
    features, output_branch = model.predict(img)
    if return_features == False:
        return np.argmax(output_branch), np.max(output_branch)
    else:
        return features.reshape(-1), [np.argmax(output_branch), np.max(output_branch)]


def predict_batch(model, imgs, return_features=True):
    batch = len(imgs)
    imgs_predict = []
    for img in imgs:
        height, width, _ = img.shape
        if height > width:
            img = np.pad(img, [[0, 0], [(height - width) // 2, (height - width) // 2], [0, 0]], mode='constant',
                         constant_values=255)
        else:
            img = np.pad(img, [[(width - height) // 2, (width - height) // 2], [0, 0], [0, 0]], mode='constant',
                         constant_values=255)
        img = cv2.resize(img, (96, 96))
        imgs_predict.append(img)
    imgs_predict = np.array(imgs_predict)
    imgs_predict = imgs_predict.astype('float') / 255.
    features, output_branch = model.predict(imgs_predict)
    if return_features:
        return features.reshape(batch, -1), [np.argmax(output_branch, axis=0), np.max(output_branch, axis=0)]
    else:
        return [np.argmax(output_branch, axis=1).reshape(batch, -1), np.max(output_branch, axis=1).reshape(batch, -1)]


def predict_merge_model(model, img):
    height, width, _ = img.shape
    if height > width:
        delta = height - width
        img_ = np.pad(img, [[0, 0], [delta // 2, delta // 2], [0, 0]], mode='constant', constant_values=255)
    else:
        delta = width - height
        img_ = np.pad(img, [[delta // 2, delta // 2], [0, 0], [0, 0]], mode='constant', constant_values=255)
    img_ = cv2.resize(img_, (224, 224))
    img_ = np.transpose(img_,[2,0,1])
    img_ = np.expand_dims(img_, axis=0)
    img_ = img_.astype('float') / 255.
    img_ = torch.Tensor(img_).to('cuda:0')
    output = model.predict(img_).detach().cpu().numpy()
    # return output
    return np.argmax(output,axis=-1), np.max(output,axis=-1)
