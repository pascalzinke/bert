import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on " + torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device
