import string
import argparse
import sys
sys.path.append('textSpotting/textRecognition' )

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from dataset import RawDataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import cv2 
import numpy as np

def crop_with_padding(img, pts):
    pts_ = pts.reshape((-1, 2)).astype(int)

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

    return dst2

def text_recog(img, boxes, recog_model):
    temp = {'boxes': None,'pil_img': None, 'text': None}
    result = []

    for poly in boxes:
        x,y,w,h = cv2.boundingRect(poly)
        if h > 10 and w > 10:
            save_dict = temp.copy()
            save_dict['boxes'] = poly
            save_dict['pil_img'] = Image.fromarray(cv2.cvtColor(crop_with_padding(img, poly), cv2.COLOR_BGR2RGB))
            result.append(save_dict)

    all_crop_img = [item['pil_img'] for item in result]
    all_crop_imgs = [image.convert('L') for image in all_crop_img]
    result_text_recog = recog_model.infer(images=all_crop_imgs)

    for count, item in enumerate(result_text_recog):
        result[count]['text'] = item[0]

    return result

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class InferDataset(Dataset):
    def __init__(self, images):
        self.images = images

        self.n_samples = len(images)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        return img, str(idx)


class TextRecogInferer:
    def __init__(
            self,
            opt,
    ):
        self.opt = opt
        """ model configuration """
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        self.model = model
        self.converter = converter

        self.AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    @torch.no_grad()
    def infer(self, images):
        self.model.eval()
        pred_ds = InferDataset(images=images)
        pred_dl = DataLoader(
            pred_ds,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_demo,
            pin_memory=True,
            drop_last=False,
        )

        result = []

        for image_tensors, img_idxs in pred_dl:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in self.opt.Prediction:
                preds = self.model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_idx, pred, pred_max_prob in zip(img_idxs, preds_str, preds_max_prob):
                if 'Attn' in self.opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                except Exception as e:
                    # print(pred_max_prob)
                    # print(img_idx)
                    confidence_score = 0

                result.append((pred, confidence_score))

        return result


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='best_accuracy.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=80, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=320, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default="""ầỏỉờbdúnuyỷủổrkxlqằwịộc,ẹựàề#đg;a)ĩv5iìừùâý6ễh0ọ/mỵẽ9ẵứ>ỗ"ã?ă8!ấ`ảỳf&ạ(3êửếz2ô4|ụểõ+1éởốồữậưũepỡó.ẩá“~ỹ=oệèơẻ:òs7íj”ợ%ặẳớ*ắ' tẫ-""",
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default=None, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='VGG', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default=None, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    opt = parser.parse_args()
    return opt

class default_args():
    def __init__(self,model_path):
        self.workers = 0
        self.batch_size = 192
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

if __name__ == '__main__':
    #load model
    model_path = 'text-recognition/best_accuracy.pth'
    opt = default_args(model_path)
    inferer = TextRecogInferer(opt)

    #inference
    img = cv2.imread('/content/img10.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    preds = inferer.infer(
        images=[im_pil],
    )
    print(preds)
