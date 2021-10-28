import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import time
# from ..loss import build_loss, ohem_batch, iou
from ..post_processing import pa

from tqdm import tqdm 

from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Queue

from scipy.sparse import csr_matrix

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max()+1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M[1:]]


class PA_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel,
                 loss_emb):
        super(PA_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        # self.text_loss = build_loss(loss_text)
        # self.kernel_loss = build_loss(loss_kernel)
        # self.emb_loss = build_loss(loss_emb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        # pa
        label = pa(kernels, emb)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(label, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pa_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        with_rec = hasattr(cfg.model, 'recognition_head')

        if with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if with_rec:
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
                instances[0].append(i)

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))
        if with_rec:
            outputs.update(dict(
                label=label,
                bboxes_h=bboxes_h,
                instances=instances
            ))

        return outputs

    # def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes):
    #     # output
    #     texts = out[:, 0, :, :]
    #     kernels = out[:, 1:2, :, :]
    #     embs = out[:, 2:, :, :]

    #     # text loss
    #     selected_masks = ohem_batch(texts, gt_texts, training_masks)
    #     loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
    #     iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
    #     losses = dict(
    #         loss_text=loss_text,
    #         iou_text=iou_text
    #     )

    #     # kernel loss
    #     loss_kernels = []
    #     selected_masks = gt_texts * training_masks
    #     for i in range(kernels.size(1)):
    #         kernel_i = kernels[:, i, :, :]
    #         gt_kernel_i = gt_kernels[:, i, :, :]
    #         loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
    #         loss_kernels.append(loss_kernel_i)
    #     loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
    #     iou_kernel = iou(
    #         (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
    #     losses.update(dict(
    #         loss_kernels=loss_kernels,
    #         iou_kernel=iou_kernel
    #     ))

    #     # embedding loss
    #     loss_emb = self.emb_loss(embs, gt_instances, gt_kernels[:, -1, :, :], training_masks, gt_bboxes, reduce=False)
    #     losses.update(dict(
    #         loss_emb=loss_emb
    #     ))

    #     return losses

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_single_result(self, out, img_meta, cfg):
        outputs = dict()

        # temp = out.cpu().numpy()
        # score = self.sigmoid(temp[:,0,:,:])
        # kernels = temp[:, :2, :, :] > 0
        # text_mask = kernels[:, :1, :, :]
        # kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        # emb = temp[:, 2:, :, :]
        # emb = emb * text_mask.astype(float)
        
        # score = score[0].astype(np.float32)
        # kernels = kernels[0].astype(np.uint8)
        # emb = emb[0].astype(np.float32)        
        
        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        label = pa(kernels, emb)
        # image size
        org_img_size = img_meta['org_img_size']
        img_size = img_meta['img_size']
        
        # print('org_size', org_img_size)
        # print('img_size', img_size)

        label_num = np.max(label) + 1
        label = cv2.resize(label, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
        
        # label1 = label.copy()

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))
        #################### original code ########################
        # bboxes = []
        # scores = []
        
        # for i in range(1, label_num):
        #     ind = label == i
        #     points = np.array(np.where(ind)).transpose((1, 0))

        #     if points.shape[0] < cfg.test_cfg.min_area:
        #         label[ind] = 0
        #         continue

        #     score_i = np.mean(score[ind])
        #     if score_i < cfg.test_cfg.min_score:
        #         label[ind] = 0
        #         continue

        #     if with_rec:
        #         tl = np.min(points, axis=0)
        #         br = np.max(points, axis=0) + 1
        #         bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
        #         instances[0].append(i)

        #     if cfg.test_cfg.bbox_type == 'rect':
        #         rect = cv2.minAreaRect(points[:, ::-1])
        #         bbox = cv2.boxPoints(rect) * scale
        #     elif cfg.test_cfg.bbox_type == 'poly':
        #         binary = np.zeros(label.shape, dtype='uint8')
        #         binary[ind] = 1
        #         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         bbox = contours[0] * scale

        #     bbox = bbox.astype('int32')
        #     bboxes.append(bbox.reshape(-1))
        #     scores.append(score_i)

        # bboxes0 = bboxes.copy()
        # scores0 = scores.copy()
        # time3 = time.time()
        # print('for loop time', time3-time2)
        
        
        # label = label1.copy()
        # label[label > 0] = 1
        
        # cv2.imwrite('label.jpg', label*128)
        # cv2.imwrite('label1.jpg', label1)
        # label_num = np.max(label) + 1
        # print(label_num)
        
        
        ###################### multi thread ########################
        # bboxes = []
        # scores = []

        # # num_thread = label_num -1 
        # pool = ThreadPool(processes=label_num-1)
        
        # result = []
        # for i in range(1, label_num):
        #     async_result = pool.apply_async(thread_worker, (label, score, scale, cfg.test_cfg.min_area, cfg.test_cfg.min_score, i)) # tuple of args for foo
        #     result.append(async_result)
        
        # for res in result:
        #     box = res.get()[0]
        #     if np.any(box != None):
        #         bboxes.append(box)
        #         scores.append(res.get()[1])
        
        # print('num word', len(bboxes))
        
        # time4 = time.time()
        # print('for loop time',time4-time3)        
        
        
        ######################### sparse matrix #############################
        bboxes = []
        scores = []
        
        list_points = get_indices_sparse(label)

        for i in range(1, label_num):
            # ind = label == i
            # t51 = time.time()
            # points = np.array(np.where(ind)).transpose((1, 0))
            points = np.array(list_points[i-1]).transpose((1, 0))
            ind = list_points[i-1]
            # t6 = time.time()
            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue
            # t7 = time.time()
            score_i = np.mean(score[ind])
            # t8 = time.time()
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue
            # t9 = time.time()
            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale
                
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)                
            
        
        # time4 = time.time()
        # print('for loop time',time4-time3)

        # print('num word', len(bboxes))
        
        
        ####################### merge code ########################
        # bboxes = []
        # scores = []
        
        # list_points = get_indices_sparse(label)
        
        # def thread_worker_inner(i):
        #     points = np.array(list_points[i-1]).transpose((1, 0))
        #     ind = list_points[i-1]

        #     if points.shape[0] < cfg.test_cfg.min_area:
        #         label[ind] = 0
        #         return (None, None)

        #     score_i = np.mean(score[ind])
        #     if score_i < cfg.test_cfg.min_score:
        #         label[ind] = 0
        #         return (None, None)
            
        #     rect = cv2.minAreaRect(points[:, ::-1])
        #     bbox = cv2.boxPoints(rect) * scale

        #     bbox = bbox.astype('int32')
        #     return (bbox.reshape(-1), score_i)
        
        # # num_thread = label_num -1 
        # pool = ThreadPool(processes=label_num-1)
        
        # result = []
        # for i in range(1, label_num):
        #     # async_result = pool.apply_async(thread_worker0, (list_points, score, scale, cfg.test_cfg.min_area, cfg.test_cfg.min_score, i))
        #     async_result = pool.apply_async(thread_worker_inner, (i,))
        #     result.append(async_result)
        
        # for res in result:
        #     box = res.get()[0]
        #     if np.any(box != None):
        #         bboxes.append(box)
        #         scores.append(res.get()[1])
        
        # print('num word', len(bboxes))
        
        # time4 = time.time()
        # print('for loop time',time4-time3)           
        
        
        ################# multi process #######################
        # bboxes = []
        # scores = []
        
        # list_points = get_indices_sparse(label)
        
        # q = Queue()
        
        # result = []
        # for i in range(1, label_num):
        #     p = Process(target=process_worker, args=(list_points, score, scale, cfg.test_cfg.min_area, cfg.test_cfg.min_score, i, q, )) # tuple of args for foo
        #     p.start()
        
        # for i in result:
        #     box, score = q.get()
        #     if np.any(box != None):
        #         bboxes.append(box)
        #         scores.append(score)
        
        # print('num word', len(bboxes))
        
        # time4 = time.time()
        # print('for loop time',time4-time3)    
        
        
        
        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))

        return outputs
    
    
    
def thread_worker(label, score, scale, min_area, min_score, i):
    ind = label == i
    points = np.array(np.where(ind)).transpose((1, 0))

    if points.shape[0] < min_area:
        label[ind] = 0
        return (None, None)

    score_i = np.mean(score[ind])
    if score_i < min_score:
        label[ind] = 0
        return (None, None)
    
    rect = cv2.minAreaRect(points[:, ::-1])
    bbox = cv2.boxPoints(rect) * scale

    bbox = bbox.astype('int32')
    return (bbox.reshape(-1), score_i)

def thread_worker0(list_points, score, scale, min_area, min_score, i):
    points = np.array(list_points[i-1]).transpose((1, 0))
    ind = list_points[i-1]

    if points.shape[0] < min_area:
        label[ind] = 0
        return (None, None)

    score_i = np.mean(score[ind])
    if score_i < min_score:
        label[ind] = 0
        return (None, None)
    
    rect = cv2.minAreaRect(points[:, ::-1])
    bbox = cv2.boxPoints(rect) * scale

    bbox = bbox.astype('int32')
    return (bbox.reshape(-1), score_i)

def process_worker(list_points, score, scale, min_area, min_score, i, q):
    points = np.array(list_points[i-1]).transpose((1, 0))
    ind = list_points[i-1]

    if points.shape[0] < min_area:
        label[ind] = 0
        return (None, None)

    score_i = np.mean(score[ind])
    if score_i < min_score:
        label[ind] = 0
        return (None, None)
    
    rect = cv2.minAreaRect(points[:, ::-1])
    bbox = cv2.boxPoints(rect) * scale

    bbox = bbox.astype('int32')
    # return (bbox.reshape(-1), score_i)
    q.put((bbox.reshape(-1), score_i))