import torch
import logging 
import platform

LOGGER = logging.getLogger('__main__.'+__name__)

def select_device(device='',model_name=''):
    s = f'{model_name.upper()} 💥💥💥💥💥  torch {torch.__version__}'
    device = str(device).lower().replace('cuda:','').strip()
    cpu = device=='cpu'
    cuda = not cpu and torch.cuda.is_available()
    devices = device.split(',') if device else '0'
    if cuda:
        for i,d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    LOGGER.info(s.encode().decode('ascii','ignore') if platform.system()=='Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')
    
def loadingImageNetWeight(model,name):
    import io
    import requests
    model_urls = {
        'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }
    if hasattr(model,'name'):
        name = getattr(model,'name')
    assert name in model_urls, f"please model.name must be in list {list(model_urls.keys())}"
    imagenet_state_dict = torch.load(io.BytesIO(requests.get(model_urls[name]).content))
    my_state_dict = model.state_dict()
    temp = {}
    for k,v in imagenet_state_dict.items():
        if k in my_state_dict:
            if v.shape==my_state_dict.get(k).shape:
                temp[k]=v
    my_state_dict.update(temp)
    model.load_state_dict(my_state_dict)
    return model