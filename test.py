import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html


def main():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    if ',' in opt.which_epoch:
        range_str = list(map(int, str(opt.which_epoch).split(',')))
        first_epoch, last_epoch = range_str[:2]
        step = range_str[2] if len(range_str) == 3 else 1
        res_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test') if opt.results_dir is None else opt.results_dir
        for epoch in range(first_epoch, last_epoch, step):
            opt.which_epoch = epoch
            opt.results_dir = os.path.join(res_dir, 'epoch_%d' % epoch)
            test(opt)
    else:
        test(opt)


def test(opt):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create webpage
    web_dir = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch)) \
        if opt.results_dir is None else opt.results_dir
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()

if __name__ == '__main__':
    main()
