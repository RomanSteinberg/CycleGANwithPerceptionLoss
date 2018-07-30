import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html, util


def main():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    tester = Tester(opt)

    if ',' in opt.which_epoch:
        range_str = list(map(int, str(opt.which_epoch).split(',')))
        first_epoch, last_epoch = range_str[:2]
        step = range_str[2] if len(range_str) == 3 else 1
        res_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test') if opt.results_dir is None else opt.results_dir

        for epoch in range(first_epoch, last_epoch, step):
            opt.which_epoch = epoch
            opt.results_dir = os.path.join(res_dir, 'epoch_%d' % epoch)
            tester.run(opt)
    else:
        tester.run(opt)


class Tester:
    def __init__(self, opt):
        data_loader = CreateDataLoader(opt)
        self.dataset = data_loader.load_data()

    def run(self, opt):
        self.model = create_model(opt)
        self.visualizer = Visualizer(opt)
        # create webpage
        web_dir = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch)) \
            if opt.results_dir is None else opt.results_dir
        util.mkdirs(web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        # test
        for i, data in enumerate(self.dataset):
            if i >= opt.how_many:
                break
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            if opt.verbosity >=2:
                print('process image... %s' % img_path)
            self.visualizer.save_images(webpage, visuals, img_path)

        if opt.verbosity >= 1:
            print('all images were processed')
        webpage.save()

if __name__ == '__main__':
    main()
