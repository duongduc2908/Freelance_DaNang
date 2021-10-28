# import models
from .fpem_v1 import FPEM_v1


def build_neck(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    # neck = models.neck.__dict__[cfg.type](**param)
    neck = FPEM_v1(**param)

    return neck
