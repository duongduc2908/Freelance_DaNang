import torch 
import yaml 
import pandas as pd
import argparse
from utils.torch_utils import select_device,loadingImageNetWeight
from utils.dataset import LoadImagesAndLabels,preprocess
from utils.general import EarlyStoping,visualize
from utils.callbacks import CallBack
from tqdm import tqdm 
import sklearn.metrics
from models.mobilenetv2 import MobileNetV2
import os
import numpy as np
import logging 
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.basicConfig()
def train(opt):
    softmax = torch.nn.Softmax(1)
    os.makedirs(opt.save_dir, exist_ok=True)
    df_train = pd.read_csv(opt.train_csv)
    df_val = pd.read_csv(opt.val_csv)
    if 1: 
        df_train = df_train[:100]
        df_val = df_val[:100]
    device = select_device(opt.device,model_name=opt.model_name)
    visualize(df_train,classes=opt.classes,save_dir=opt.save_dir,dataset_name='train')
    visualize(df_val,classes=opt.classes,save_dir=opt.save_dir,dataset_name='val')
    
    cuda = device.type != 'cpu'
    ds_train = LoadImagesAndLabels(df_train,
                                data_folder=opt.DATA_FOLDER,
                                img_size = opt.img_size,
                                padding = opt.padding,
                                preprocess=preprocess,
                                augment=True,
                                augment_params=opt.augment_params)

    ds_val = LoadImagesAndLabels(df_val,
                                data_folder=opt.DATA_FOLDER,
                                img_size = opt.img_size,
                                padding= opt.padding,
                                preprocess=preprocess,
                                augment=False)
    trainLoader = torch.utils.data.DataLoader(ds_train,
                                             batch_size=opt.batch_size,
                                            shuffle=True,)
                                            # num_workers=8)
    valLoader = torch.utils.data.DataLoader(ds_val,
                                            batch_size = opt.batch_size,
                                            shuffle=False,)
                                            # num_workers=8)
    loader = {'train': trainLoader,
              'val'  : valLoader}
    callback = CallBack(opt.save_dir)
    # init model
    model = MobileNetV2(opt.nc)

    loss_train_log = []
    loss_val_log = []
    best_fitness,best_epoch = 0,0
    start_epoch = 0
    if os.path.isfile(opt.weights):                    # load from checkpoint
        LOGGER.info(f'loading pretrain from {opt.weights}')
        ckpt_load = torch.load(opt.weights)
        model.load_state_dict(ckpt_load['state_dict'])
        if opt.continue_training:
            start_epoch = ckpt_load['epoch'] + 1
            best_epoch = ckpt_load['best_epoch']
            loss_train_log = ckpt_load['loss_train_log']
            loss_val_log = ckpt_load['loss_val_log']
            fitness = ckpt_load['fitness']
            best_fitness = ckpt_load['best_fitness']
            LOGGER.info('resume training from last checkpoint')
    else:                                               #load from ImagesNet weight
        LOGGER.info(f"weight path : {opt.weights} does'nt exist, ImagesnNet weight will be loaded ")
        model = loadingImageNetWeight(model,name=opt.model_name)
       
    model = model.to(device)

    # optimier
    g0, g1, g2 = [], [], [] #params group  #g0 - BatchNorm, g1 - weight, g2 - bias
    for module in model.modules():
        if hasattr(module,'bias') and isinstance(module.bias, torch.nn.Parameter):
            g2.append(module.bias)
        if isinstance(module, torch.nn.BatchNorm2d):
            g0.append(module.weight)
        elif hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            g1.append(module.weight)
    if hasattr(opt,'adam'):
        optimizer = torch.optim.Adam(g0, lr=opt.hyp['lr0'], betas=opt.hyp['momentum'])
    else:
        optimizer = torch.optim.SGD(g0, lr=opt.hyp['lr0'], momentum=opt.hyp['momentum'])
    optimizer.add_param_group({'params':g1, 'weight_decay': opt.hyp['weight_decay']})
    optimizer.add_param_group({'params':g2})
    del g0,g1,g2

    #loss function
    # class_weights = torch.Tensor(opt.class_weights).to(device) if hasattr(opt,'class_weights') else None
    # criterior = torch.nn.CrossEntropyLoss(weight=class_weights)
    criteriors = list()
    for label_name in opt.classes.items():
        if hasattr(opt.class_weights,label_name):
            class_weights = torch.Tensor(opt.class_weights[label_name])
            class_weights = class_weights/class_weights.sum()
            class_weights = class_weights.to(device)
        else: 
            class_weights = None
        criteriors.append(torch.nn.CrossEntropyLoss(weight=class_weights))
        

    if not isinstance(opt.task_weights,list):
        task_weights = [opt.task_weights]
    
    
    # if opt.linear_lr:
    #     lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - opt.hyp['lrf']) + opt.hyp['lrf']  # linear
    # else:
    #     # lf = one_cycle(1, opt.hyp['lrf'], epochs)
    #     pass

    stopper = EarlyStoping(best_fitness=best_fitness, best_epoch=best_epoch, patience=opt.patience)
    
    pbar_epoch = tqdm(range(start_epoch,opt.epochs),total=opt.epochs,initial=start_epoch)
    
    for epoch in pbar_epoch:
        # training phase.
        if epoch!=0 and opt.sampling_balance_data:
            loader['train'].dataset.on_epoch_end(n=opt.sampling_balance_data)
        
        model.train()
        pbar = enumerate(loader['train'])
        nb = len(loader['train'])
        epoch_loss = 0.0
        warmup_iteration = max(opt.hyp['warmup_epochs']*nb,1000)
        for i, (imgs, labels, _) in tqdm(pbar,total=nb,desc='training',leave=False):
            ni = i + nb * epoch
            imgs = imgs.to(device)
            labels = [label.to(device) for label in labels]
            # Warm-up training
            if ni <= warmup_iteration:
                xi = [0, warmup_iteration]  # x interp
                # accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    #bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [opt.hyp['warmup_bias_lr'] if j == 2 else 0.0, opt.hyp['lr0']]) #x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [opt.hyp['warmup_momentum'], opt.hyp['momentum']])
            
            
            with torch.set_grad_enabled(cuda):
                preds = model(imgs)
                # loss = criterior(preds,labels)
                loss = 0
                for index,criterior in enumerate(criteriors):
                    loss += opt.task_weight[index]*criterior(preds[index],labels[index])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                epoch_loss += loss * imgs.size(0)
        epoch_loss = epoch_loss/len(loader['train'].dataset)
        loss_train_log.append(epoch_loss)

        # evaluate phase.
        model.eval()
        epoch_loss = 0
        
        y_trues = {}
        y_preds = {}
        for index,label_name in enumerate(list(opt.classes.keys())):
            y_trues[label_name] = []
            y_preds[label_name] = []

        with torch.no_grad():
            for i, (imgs, labels, _) in tqdm(enumerate(loader['val']),total=len(loader['val']),desc='evaluating',leave=False):
                imgs = imgs.to(device)
                labels = [label.to(device) for label in labels]
                preds = model(imgs)
                # preds_after_softmax = [sofmax(x) for x in preds]
                # for index,label_name in enumerate(list(opt.classes.keys())):
                #     y_preds[label_name].append(preds_after_softmax[index].detach().cpu().numpy())
                #     y_trues[label_name].append(labels[index].detach().cpu().numpy())
                loss = 0
                for index,criterior in enumerate(criteriors):
                    loss += opt.task_weight[index]*criterior(preds[index],labels[index])                      
                epoch_loss += loss * imgs.size(0)
        epoch_loss = epoch_loss/len(loader['val'].dataset)
        loss_val_log.append(epoch_loss)

        # y_true = np.concatenate(y_true, axis=0)
        # for index,label_name in enumerate(list(opt.classes.keys())):
        #     y_true = 
        

        
        
        # fi - macro avg accuracy

        # fi = sklearn.metrics.classification_report(y_true,y_pred,digits=4,zero_division=1)
        # fi = fi.split('\n')[-3].split()[-2]
        # fi = float(fi)

        fi = epoch_loss.item()

        if stopper(epoch,fi):   #if EarlyStopping condition meet
            break
        if epoch==stopper.best_epoch:
            ckpt_best = { 
                            'state_dict':model.state_dict(),
                            'best_fitness': stopper.best_fitness,
                            'fitness': fi, 
                            'epoch':epoch,
                            'best_epoch': epoch,
                            'loss_train_log': loss_train_log,                  
                            'loss_val_log': loss_val_log
                        }
            torch.save(ckpt_best,os.path.join(opt.save_dir,'best.pt'))
        ckpt_last = { 
                        'state_dict':model.state_dict(),
                        'best_fitness': stopper.best_fitness,
                        'fitness': fi,  
                        'epoch': epoch,
                        'best_epoch': stopper.best_epoch,
                        'loss_train_log': loss_train_log,                  
                        'loss_val_log': loss_val_log
                    }
        torch.save(ckpt_last,os.path.join(opt.save_dir,'last.pt'))   
        callback(loss_train_log[-1],loss_val_log[-1],epoch)   
        # _ = os.system('clear')
def parse_opt(know=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',help = 'weight path')
    parser.add_argument('--cfg',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/default/train_config.yaml')
    parser.add_argument('--data',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/default/data_config.yaml')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30, help='patience epoch for EarlyStopping')
    parser.add_argument('--save_dir', type=str, default='', help='save training result')
    parser.add_argument('--task_weights', type=list, default=1, help='weighted for each task while computing loss')
    opt = parser.parse_known_args()[0] if know else parser.parse_arg()
    return opt 

if __name__ =='__main__': 
    opt = parse_opt(True)
    with open(opt.cfg) as f:
        cfg = yaml.safe_load(f)
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    for k,v in cfg.items():
        setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    assert isinstance(opt.classes,dict), "Invalid format of classes in data_config.yaml"
    assert len(opt.task_weights) == len(opt.classes), "task weight should has the same length with classes"
    # print(opt)
    train(opt)
