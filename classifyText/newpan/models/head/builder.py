# import models
from .pa_head import PA_Head


def build_head(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    # head = models.head.__dict__[cfg.type](**param)
    head = PA_Head(**param)

    return head
