import os
import torch.nn as nn
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from time import gmtime, strftime


class TBLogger:
    def __init__(self, opt, *args):
        self._writer = {}
        for name in args:
            log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboardX', name)
            date_now = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            self._writer[name] = SummaryWriter(log_dir=log_dir, filename_suffix=f'.{name}.{date_now}')
        layout = {
            'losses': {'value losses': ['Multiline', [f'loss/{name}'for name in args]]}
        }
        self._writer[args[0]].add_custom_scalars(layout)

    def log_graph(self, model, network_name, input_net):
        self._writer[network_name].add_graph(getattr(model, network_name), input_net)

    def log_loss(self, model, network_name, iteration):
        error_key = network_name.split('t')[1]
        error = model.get_current_errors()[error_key]
        self._writer[network_name].add_scalar(f'loss/{network_name}', error.data, global_step=iteration)

    def log_gradients(self, model, network_name, iteration):
        i = 0
        for name, param in list(getattr(model, network_name).named_parameters())[::-1]:
            split_name = name.split('.')
            if split_name[-1] == 'weight':
                name = '.'.join(x for x in split_name[0:-1]) + '.gradients'
                self._writer[network_name].add_histogram(name, param.grad, iteration, bins='doane')
                self._writer[network_name].add_histogram(f'hist_gradients/{network_name}_{iteration}', param.grad,
                                                         global_step=i)
                self._writer[network_name].add_histogram(f'hist_weights/{network_name}_{iteration}', param.data,
                                                         global_step=i)
                i += 1

    def log_weights_spectrum(self, model, network_name, iteration):
        i = 0
        for layer in getattr(model, network_name).modules():
            if isinstance(layer, nn.Conv2d):
                conv_weights = list(layer.parameters())[0].cpu().detach().numpy()
                eigenvalues = LA.eigvals(conv_weights)
                spectrum_variance = np.var(eigenvalues)
                self._writer[network_name].add_scalar(f'spectrum_variance/{network_name}_conv{i}', spectrum_variance, global_step=iteration)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                X = [x.real for x in eigenvalues.flatten()]
                Y = [y.imag for y in eigenvalues.flatten()]
                ax.scatter(X, Y)
                ax.grid()
                ax.set_ylabel('Imaginary')
                ax.set_xlabel('Real')
                self._writer[network_name].add_figure(f'spectrum_scatter/{network_name}_conv{i}_{iteration}', fig, global_step=iteration)
                plt.clf()

                i += 1

    def __del__(self):
        for key in self._writer:
            self._writer[key].close()
