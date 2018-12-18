from __future__ import print_function

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import inspect, re
import os
import collections

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from argparse import Namespace


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    mkdirs(os.path.dirname(image_path))
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_switch_norm_hists(model, epoch, opt):
    """
    Saves 3D histograms of distributions of statistics (mean and variance) of switch norm layers.

    Args:
        model(nn.Module): PyTorch model.
        epoch(int): number of current epoch.
        opt(Namespace): command line options.
    """
    statistics = ['switch_norm_mean_weight', 'switch_norm_var_weight']
    for stat in statistics:
        norm_layers_names = list(filter(lambda x: stat in x, model.state_dict().keys()))
        norm_layers_values = np.array([np.array(model.state_dict()[name]) for name in norm_layers_names])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Grid upon which hists will be built.
        x_data, y_data = np.meshgrid(np.arange(norm_layers_values.shape[1]),
                                     np.arange(norm_layers_values.shape[0]))

        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = norm_layers_values.flatten()

        space_between_different_norms = 0.7
        space_between_weights_on_each_epoch = 0.9
        ax.bar3d(x=x_data,
                 y=y_data,
                 z=np.zeros(len(z_data)),
                 dx=space_between_different_norms,
                 dy=space_between_weights_on_each_epoch,
                 dz=z_data)

        savefig_folder = os.path.join(opt.checkpoints_dir, opt.name, 'normalizations')
        os.makedirs(savefig_folder, exist_ok=True)

        stat_type = stat.split('_')[2]
        savefig_fname = os.path.join(savefig_folder, f'{stat_type}_{epoch}.png')
        fig.savefig(savefig_fname)
