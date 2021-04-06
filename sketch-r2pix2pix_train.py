import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images, save_mean_and_var_images
from util import html
from torch.utils.tensorboard import SummaryWriter
import torchvision
import random
import os

if __name__ == '__main__':

    writer = SummaryWriter() #create tensorboard summary writer

    opt = TrainOptions().parse()   # get training options

    #TODO: replace with SketchR2CNN dataset
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.


        #look to see if changing the dataset makes a difference here
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            #get a new vector sketch and the image it is paired with
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            loss_D, individual_D_losses, loss_G, individual_G_losses = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            writer.add_scalar('Discriminator Loss',
                            loss_D,
                            epoch * len(dataset) + epoch_iter)
            writer.add_scalar('Discriminator Fake Image Loss',
                            individual_D_losses[0],
                            epoch * len(dataset) + epoch_iter)
            writer.add_scalar('Discriminator Real Image Loss',
                            individual_D_losses[1],
                            epoch * len(dataset) + epoch_iter)
            if len(individual_D_losses) == 3:
                writer.add_scalar('Discriminator Classification Loss', # category loss
                                individual_D_losses[2],
                                epoch * len(dataset) + epoch_iter)
            
            writer.add_scalar('Generator Loss',
                            loss_G,
                            epoch * len(dataset) + epoch_iter)
            writer.add_scalar('GAN Loss',
                            individual_G_losses[0],
                            epoch * len(dataset) + epoch_iter)
            writer.add_scalar('L1 Loss',
                            individual_G_losses[1],
                            epoch * len(dataset) + epoch_iter)
            if len(individual_D_losses) == 3:
                writer.add_scalar('Generator Classification Loss', # category loss
                                individual_G_losses[2],
                                epoch * len(dataset) + epoch_iter)

            rnn_param_grads, g_param_grads = model.get_param_grads()

            writer.add_scalar(f'RNN parameter gradient mean',
                            rnn_param_grads,
                            epoch * len(dataset) + epoch_iter)

            writer.add_scalar(f'G parameter gradient mean',
                            g_param_grads,
                            epoch * len(dataset) + epoch_iter)


            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


    #test the model - saves images in results directory
    rand_target = round(100 / dataset_size)
    web_dir = os.path.join('./results', opt.name)  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = f'{web_dir}'
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, f'{web_dir}')

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print('processing (%04d)-th image... %s' % (i, img_path))
        for label, im in visuals.items():
            if label == 'real_A':
                save_mean_and_var_images(webpage, img_path, im)
        save_images(webpage, visuals, img_path, aspect_ratio=1.0, width=opt.display_winsize)
    webpage.save()  # save the HTML