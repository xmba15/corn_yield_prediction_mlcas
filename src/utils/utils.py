import pydoc
import random

import numpy as np
import timm
import torch

__all__ = (
    "get_object_from_dict",
    "fix_seed",
    "worker_init_fn",
    "get_extractor_in_features",
)


def get_object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_extractor_in_features(
    encoder_name,
    in_chans=3,
):
    model = timm.create_model(
        encoder_name,
        in_chans=in_chans,
    )
    return model.get_classifier().in_features
