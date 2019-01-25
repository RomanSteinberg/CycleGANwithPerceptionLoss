import time
import torch
import numpy as np

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import save_switch_norm_hists
from util.visualizer import Visualizer
from util.tensorboardX_logger import TBLogger
from hyperdash import Experiment

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
tb_logger = TBLogger(opt, loss_arr, 'netG_A', 'netG_B', 'netD_A', 'netD_B')
model = create_model(opt)
visualizer = Visualizer(opt, loss_arr)
tb_logger.log_graph(model, 'netG_A', model.input_A)
tb_logger.log_graph(model, 'netG_B', model.input_B)
tb_logger.log_graph(model, 'netD_A', model.input_A)
tb_logger.log_graph(model, 'netD_B', model.input_B)

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

        tb_logger.log_gradients(model, 'netG_A', i)
        tb_logger.log_gradients(model, 'netG_B', i)
        tb_logger.log_gradients(model, 'netD_A', i)
        tb_logger.log_gradients(model, 'netD_B', i)

        if total_steps % opt.display_freq == 0:
            # turned off due to opt.display_id==0
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, dataset_size)

            tb_logger.log_loss(model, 'netG_A', i)
            tb_logger.log_loss(model, 'netG_B', i)
            tb_logger.log_loss(model, 'netD_A', i)
            tb_logger.log_loss(model, 'netD_B', i)

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

print('End of training. Time Taken: %d sec' % (time.time() - train_start_time))
