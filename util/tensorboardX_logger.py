import os

from tensorboardX import SummaryWriter


class TBLogger:
    def __init__(self, opt, *args):
        self.writer = {}
        for name in args:
            log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboardX', name)
            self.writer[name] = SummaryWriter(log_dir=log_dir, filename_suffix=f'.{name}')
        layout = {
            'losses': {'value losses': ['Multiline', [f'loss/{name}'for name in args]]}
        }
        self.writer[args[0]].add_custom_scalars(layout)

    def log_graph(self, model, network_name, input_net):
        self.writer[network_name].add_graph(getattr(model, network_name), input_net)

    def log_loss(self, model, network_name, iteration):
        error_key = network_name.split('t')[1]
        error = model.get_current_errors()[error_key]
        self.writer[network_name].add_scalar(f'loss/{network_name}', error.data, global_step=iteration)

    def log_gradients(self, model, network_name, iteration):
        # network_gradients = []
        for name, param in getattr(model, network_name).named_parameters():
            split_name = name.split('.')
            if split_name[-1] == 'weight':
                self.writer[network_name].add_histogram('.'.join(x for x in split_name[0:-1])+'.gradients', param.grad, iteration, bins='doane')
                # network_gradients.append(param.grad.cpu().numpy())

    def __del__(self):
        for key in self.writer:
            self.writer[key].close()
