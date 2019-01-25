import os

from tensorboardX import SummaryWriter


class TBLogger:
    def __init__(self, opt, *args):
        self._writer = {}
        for name in args:
            log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboardX', name)
            self._writer[name] = SummaryWriter(log_dir=log_dir, filename_suffix=f'.{name}')

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
                self._writer[network_name].add_histogram(f'hist/{network_name}_{iteration}', param.grad, global_step=i)
                i += 1

    def __del__(self):
        for key in self._writer:
            self._writer[key].close()
