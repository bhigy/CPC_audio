#!/usr/bin/env python3

"""
Example of a generic Mixout implementation. (Lee et al., 2019).
https://arxiv.org/abs/1909.11299
Implementation by Stephen Roller (https://stephenroller.com).
Updated 2020-02-10 to include 1/(1 - p) correction term. Thanks to
Cheolhyoung Lee for making this correction.
Example output:
$ python mixout.py
parameter: 0.weight   Vanilla distance: 0.00239  Mixout distance: 0.00128
parameter: 0.bias     Vanilla distance: 0.000191  Mixout distance: 5.8e-05
parameter: 2.weight   Vanilla distance: 0.000494  Mixout distance: 0.000258
parameter: 2.bias     Vanilla distance: 1.75e-05  Mixout distance: 1.01e-05
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def MixoutWrapper(module: nn.Module, p: float = 0.9):
    """
    Implementation of Mixout (https://arxiv.org/abs/1909.11299).
    by Stephen Roller (https://stephenroller.com)
    Modified from original version (https://gist.github.com/stephenroller/f45a372e231825f9f5578e9e705f4e95)
    to handle multiple-GPUs.
    Use with:
    >>> mixout_model = model.apply(MixoutWrapper).
    """
    # duplicate all the parameters, making copies of them and freezing them
    module._names = []
    module._params_orig = dict()
    _params_learned = nn.ParameterDict()
    for n, q in list(module.named_parameters(recurse=False)):
        c = torch.tensor(q.clone())
        # c.requires_grad = False
        module._params_orig[n] = c
        _params_learned[n] = q
        module._names.append(n)
        delattr(module, n)
        setattr(module, n, c)
    if module._names:
        module._params_learned = _params_learned

    def mixout(module, n):
        if module.training:
            o = module._params_orig[n]
            mask = (torch.rand_like(o) < p).type_as(o)
            # update 2020-02-
            return (
                mask * module._params_orig[n]
                + (1 - mask) * module._params_learned[n]
                - p * module._params_orig[n]
            ) / (1 - p)
        else:
            return torch.tensor(module._params_learned[n])

    def hook(module, input):
        for n in module._names:
            try:
                v = mixout(module, n)
                setattr(module, n, v)
            except:
                breakpoint()

    module.register_forward_pre_hook(hook)
    return module

def mGPUsMixoutWrapper(module: nn.Module, p: float = 0.5):
    """
    Implementation of Mixout (https://arxiv.org/abs/1909.11299).
    by Stephen Roller (https://stephenroller.com)
    Modified from original version (https://gist.github.com/stephenroller/f45a372e231825f9f5578e9e705f4e95)
    to handle multiple-GPUs by Tu Anh NGUYEN (nguyentuanh208@gmail.com).
    Use with:
    >>> mixout_model = model.apply(mGPUsMixoutWrapper).
    """
    # duplicate all the parameters, making copies of them and freezing them
    module._names = []
    module._params_orig = dict()
    _params_learned = nn.ParameterDict()
    for n, q in list(module.named_parameters(recurse=False)):
        # modified to handle multiple-GPUs
        c = q.clone()
        if module.__class__.__name__ not in ['RNN', 'GRU', 'LSTM']: # Deal with RNN classes that have flatten_parameters()
            c = c.detach()
            c.requires_grad = False
        module._params_orig[n] = c
        _params_learned[n] = q
        module._names.append(n)
        delattr(module, n)
        setattr(module, n, c)
    if module._names:
        module._params_learned = _params_learned

    def mixout(module, n):
        if module.training:
            o = module._params_orig[n]
            mask = (torch.rand_like(o) < p).type_as(o)
            # modified to handle multiple-GPUs
            combined_params = (1 - mask) * module._params_learned[n]
            with torch.no_grad():
                combined_params += mask * module._params_orig[n] - p * module._params_orig[n]
            combined_params = combined_params / (1 - p)
            return combined_params
        else:
            return module._params_learned[n]

    def hook(module, input):
        for n in module._names:
            try:
                v = mixout(module, n)
                setattr(module, n, v)
            except:
                breakpoint()

    module.register_forward_pre_hook(hook)
    return module

def get_mixout_learned_state_dict(learned_state_dict):
    converted_state_dict = OrderedDict()
    for name, value in learned_state_dict.items():
        if '_params_learned' in name:
            name = name.replace('._params_learned','')
        converted_state_dict[name] = value
    return converted_state_dict


def learn_vanilla():
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1)

    o = torch.optim.Adam(model.parameters(), 3e-4)
    for _ in range(10):
        o.zero_grad()
        x = torch.randn(16, 64)
        y = torch.ones((16), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        o.step()

    return list(model.named_parameters())


def learn_mixout():
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1)

    mixed = model.apply(MixoutWrapper)
    o = torch.optim.Adam(mixed.parameters(), 3e-4)
    for _ in range(10):
        o.zero_grad()
        x = torch.randn(16, 64)
        y = torch.ones((16), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(mixed(x), y)
        loss.backward()
        o.step()

    return list(mixed.named_parameters())


def main():
    """
    Test mixout by checking the mixout moves slower from the initial parameters
    than the vanilla implementation.
    """
    vanilla = learn_vanilla()
    mixed = learn_mixout()

    for (name, pv), (name2, pm) in zip(vanilla, mixed):
        # we expect the parameters of the mixed model to be closer to all ones
        # than the vanilla is
        vanilla_distance = ((pv - 1) ** 2).sum()
        mixed_distance = ((pm - 1) ** 2).sum()
        print(
            f"parameter: {name:10s} "
            f"Vanilla distance: {vanilla_distance:.03}  "
            f"Mixout distance: {mixed_distance:.03}"
        )
        assert mixed_distance < vanilla_distance


if __name__ == "__main__":
    main()