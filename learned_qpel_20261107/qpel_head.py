"""Shared model and ordered direction export for training and single-file encoding."""
import time

import numpy as np
import torch
from torch import nn


def model():
    return nn.Sequential(nn.Linear(27,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,18))


@torch.inference_mode()
def predict(network, folder):
    features = np.load(folder/"features.npy",allow_pickle=False)
    start = time.perf_counter()
    logits = network(torch.from_numpy(features).cuda()).reshape(-1,2,9)
    order = logits[:,:,:8].argsort(dim=-1,descending=True)
    order.byte().cpu().numpy().tofile(folder/"qpel_prior.bin")
    return time.perf_counter()-start
