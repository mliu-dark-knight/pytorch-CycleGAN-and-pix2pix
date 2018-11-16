import os
import time
from collections import OrderedDict
from copy import deepcopy

import torch.nn as nn

from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loaders = OrderedDict()
    datasets = OrderedDict()
    opt_copy = deepcopy(opt)
    for task in opt.tasks:
        opt_copy.task = task
        opt_copy.dataroot = os.path.join(opt.dataroot, task)
        data_loader = CreateDataLoader(opt_copy)
        dataset = data_loader.load_data()
        dataset_size = len(dataset)
        data_loaders[task] = data_loaders
        datasets[task] = dataset

    dataset_size = sum([len(dataset) for dataset in datasets.values()])

    model = nn.DataParallel(create_model(opt))
    model.module.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    print('Save initialzed models')
    model.module.save_networks('latest')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        next_iters = [(task, iter(dataset)) for task, dataset in datasets.items()]
        while True:
            if len(next_iters) == 0:
                break
            iters = next_iters
            next_iters = []
            for task, dataset in iters:
                try:
                    data = next(dataset)
                    next_iters.append((task, dataset))
                except StopIteration:
                    continue
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                model.module.set_input(data, task)
                model.module.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.module.get_current_visuals(), epoch, save_result, task)

                if total_steps % opt.print_freq == 0:
                    losses = model.module.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, task, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    model.module.save_networks('latest')

                iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.module.save_networks('latest')
            model.module.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.module.update_learning_rate()
