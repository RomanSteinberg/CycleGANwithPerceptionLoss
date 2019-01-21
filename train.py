import time
import torch
import numpy as np
import os

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import save_switch_norm_hists
from util.visualizer import Visualizer
from hyperdash import Experiment
from tensorboardX import SummaryWriter

opt = TrainOptions().parse()

SEED = opt.random_seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

loss_arr = ['D_A', 'G_A', 'Cyc_A', 'idt_A', 'D_B', 'G_B', 'Cyc_B', 'idt_B']
tensorboard_path = os.path.join(opt.checkpoints_dir, opt.name)
writer = SummaryWriter(tensorboard_path)
model = create_model(opt)
visualizer = Visualizer(opt, loss_arr)
writer.add_graph(model.netG_A, model.input_A)
# writer.add_graph(model.netG_B, model.input_B)

# network_params = [param.grad for param in model.netG_A.parameters()]
# print(len(network_params))

# writer.add_graph(model.netD_A, model.input_A)
# writer.add_graph(model.netD_B, model.input_B)

total_steps = 0
start_epoch = 1 if opt.which_epoch == 'latest' else int(opt.which_epoch)
end_epoch = opt.niter + opt.niter_decay
train_start_time = time.time()
for epoch in range(start_epoch, end_epoch + 1):
    epoch_start_time = time.time()
    prev_total_steps = total_steps
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - prev_total_steps

        model.set_input(data)
        model.optimize_parameters()

        network_params = []
        for name, param in model.netG_A.named_parameters():
            if name.split('.')[-1] == 'weight':
                writer.add_histogram(name, param.grad, i, bins='doane')

        if total_steps % opt.display_freq == 0:
            # turned off due to opt.display_id==0
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, dataset_size)
            for err in errors:
                writer.add_scalar(f'loss {err}', errors[err].data, global_step=i)
            # writer.add_scalars()
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    epoch_time = time.time() - epoch_start_time
    eta = (end_epoch - epoch) * (time.time() - train_start_time) / (3600 * (epoch - start_epoch + 1))
    print('End of epoch %d / %d \t Time Taken: %d sec, ETA: %f h' % (epoch, end_epoch, epoch_time, eta))

    if epoch > opt.niter:
        model.update_learning_rate()

    if opt.norm == 'switchable':
        save_switch_norm_hists(model=model, epoch=epoch, opt=opt)

writer.close()
print('End of training. Time Taken: %d sec' % (time.time() - train_start_time))
