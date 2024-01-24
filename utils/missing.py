import torch


def generate_mask(features, missing_rate, missing_type):
    if missing_type == 'Type1':
        return generate_mask1(features, missing_rate)
    elif missing_type == 'Type2':
        return generate_mask2(features, missing_rate)
    else:
        raise ValueError("Missing type {0} is not defined".format(missing_type))


def generate_mask1(features, missing_rate):
    mask = torch.rand(size=features.size())
    return mask <= missing_rate


def generate_mask2(features, missing_rate):
    node_mask = torch.rand(size=(features.size(0), 1))
    mask = (node_mask <= missing_rate).repeat(1, features.size(1))
    return mask