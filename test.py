import os
from collections import OrderedDict
from copy import deepcopy

from torch import nn

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display

    data_loaders = OrderedDict()
    datasets = OrderedDict()
    opt_copy = deepcopy(opt)
    for task in opt.tasks:
        if opt.eval_subtask is not None:
            if task != opt.eval_subtask:
                continue
        opt_copy.task = task
        opt_copy.dataroot = os.path.join(opt.dataroot, task)
        data_loader = CreateDataLoader(opt_copy)
        dataset = data_loader.load_data()
        dataset_size = len(dataset)
        data_loaders[task] = data_loaders
        datasets[task] = dataset

    model = nn.DataParallel(create_model(opt))
    model.module.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.module.eval()
    for task, dataset in datasets.items():
        print('Processing task %s' % (task))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                (opt.name, opt.phase, opt.epoch), filename='%s.html' % task)
        webpage.add_header(task)
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            model.module.set_input(data, task)
            model.module.test()
            visuals = model.module.get_current_visuals()
            img_path = model.module.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        # Save webpage for dataset
        webpage.save()
