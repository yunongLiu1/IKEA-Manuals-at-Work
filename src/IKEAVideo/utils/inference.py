import torch

def move_to_gpu(batch, device):
    # move batch[key] to device if batch[key] is a tensor
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)


def repeat_batch(batch, k):
    # if batch[key] is a tensor, repeat first dimension k times
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = torch.repeat_interleave(batch[key], k, dim=0)


def move_to_numpy(batch):
    # if batch[key] is a tensor, do the following conversions
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            # if it has gradient, detach it
            if batch[key].requires_grad:
                batch[key] = batch[key].detach()
            # if it is on gpu, move it to cpu
            if batch[key].is_cuda:
                batch[key] = batch[key].cpu()
            # convert it to numpy
            batch[key] = batch[key].numpy()
