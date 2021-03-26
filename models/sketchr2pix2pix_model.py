import torch
from .base_model import BaseModel
from . import networks
from .base_train_sketchr2cnn import SketchR2CNNTrain
from .sketchy_dataset import SketchyDataset
from data.base_dataset import get_params, get_transform
from PIL import Image
import numpy as np

#for debugging only
import random


class SketchR2Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        #sketchrcnn model
        self.sketchr2cnn = SketchR2CNNTrain() #or SketchR2CNNEval() depending on the mode
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # get param list for the Generator Optimizer
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_RNN = torch.optim.Adam(list(self.sketchr2cnn.get_rnn_params()), lr=opt.lr*20, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_RNN)
            self.optimizers.append(self.optimizer_D)
        
        self.svg_dataset = SketchyDataset('datasets/sketchy.pkl', 'train')

        #dictionary of sketchy_gan categories
        self.category_dict = {
        'n04398044': ('teapot', 0), 'n02503517': ('elephant', 1), 'n03147509': ('cup', 2), 'n02976123': ('knife', 3), 'n02980441': ('castle', 4),
        'n02109525': ('dog', 5), 'n02374451': ('horse', 6), 'n07697537': ('hotdog', 7), 'n03891251': ('bench', 8), 'n02948072': ('candle', 9),
        'n04148054': ('scissors', 10), 'n04090263': ('rifle', 11), 'n07739125': ('apple', 12), 'n02691156': ('airplane', 13),
        'n02439033': ('giraffe', 14), 'n02121620': ('cat', 15), 'n12998815': ('mushroom', 16), 'n03544143': ('hourglass', 17),
        'n01887787': ('cow', 18), 'n07873807': ('pizza', 19), 'n07695742': ('pretzel', 20), 'n02395406': ('pig', 21), 'n02738535': ('chair', 22),
        'n07745940': ('strawberry', 23), 'n07753592': ('banana', 24), 'n09472597': ('volcano', 25), 'n01770393': ('scorpion', 26),
        'n02219486': ('ant', 27), 'n02206856': ('bee', 28), 'n04256520': ('couch', 29), 'n02317335': ('starfish', 30), 'n02129165': ('lion', 31),
        'n02346627': ('hedgehog', 32), 'n02950826': ('cannon', 33), 'n02391049': ('zebra', 34), 'n09288635': ('geyser', 35), 'n02411705': ('sheep', 36),
        'n04389033': ('tank', 37), 'n01910747': ('jellyfish', 38), 'n03790512': ('motorcycle', 39), 'n07753275': ('pineapple', 40),
        'n03633091': ('spoon', 41), 'n02834778': ('bicycle', 42), 'n03481172': ('hammer', 43), 'n02131653': ('bear', 44), 'n02129604': ('tiger', 45),
        'n01944390': ('snail', 46), 'n03028079': ('church', 47), 'n02824448': ('bell', 48)
        }
    

    def set_input(self, input):
        self.real_B = input['B'].to(self.device)
        #print(f'real B dimensions {self.real_B.shape}')
        self.AB_path = input['A_paths']

        #added to make the test code to save some images work
        self.image_paths = input['A_paths']

        #get list of real As
        self.real_As = []
        search_filename = self.AB_path[0].split('/')[-1]
        search_filename = search_filename[:-4]

        search_category = search_filename.split('_')[0]

        #a tensor containing just one value which is the category index
        #correct_category = torch.LongTensor([self.category_dict[search_category][1]]).to(self.device)
        correct_category = self.category_dict[search_category][1]
        self.correct_category = torch.LongTensor([correct_category]).to(self.device)
        fnames = self.svg_dataset.get_fnames()
        svg_file_index = None

        indices = []
        for i, fname in enumerate(fnames):
            if fname.startswith(search_filename):
                indices.append(i)
        
        if len(indices) is None:
            raise Exception(f'could not find matching file for {search_filename}')
        
        for index in indices:
            svg_data = self.svg_dataset[index]

            t = self.sketchr2cnn.get_image(svg_data)
            t = t * 255
            t = t.unsqueeze(0)
            t = torch.nn.functional.interpolate(t,size=(256,256), mode='bilinear')     
            self.real_As.append(t)

    def forward(self):
        #print(f'real A dimensions {self.real_A.shape}')
        self.real_A = self.real_As[-1]
        self.fake_B = self.netG(self.real_A)  # G(A)
        del self.real_As[-1]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake, pred_category = self.netD(fake_AB.detach())

        ce_loss = torch.nn.CrossEntropyLoss()

        #self.nll_loss = torch.nn.functional.nll_loss(pred_category, self.correct_category)

        #pred_category_flattened = pred_category.sum(3).sum(2)
        #print(pred_category_flattened.shape)
        #self.nll_loss = ce_loss(pred_category_flattened, self.correct_category)

        self.nll_loss = ce_loss(pred_category, self.correct_category)

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)[0]
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.nll_loss) * 0.5
        self.loss_D.backward(retain_graph=True)
        return self.loss_D

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #add code for SketchR2CNN training here

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)[0]
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.nll_loss #add nll loss too
        self.loss_G.backward()
        return(self.loss_G)

    def optimize_parameters(self):
        while len(self.real_As) > 0:
            self.forward()                   # compute fake images: G(A)
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            loss_D = self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.optimizer_RNN.zero_grad()        # set G's gradients to zero
            loss_G  = self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
            self.optimizer_RNN.step()
            return loss_D, loss_G


    def get_param_grads(self):
        '''
        return the mean value of each parameter gradient for the RNN
        '''
        rnn_grads = []
        for param in self.sketchr2cnn.get_rnn_params():
            rnn_grads.append(torch.mean(param.grad.view(-1)))
            break

        g_grads = []
        for param in self.netG.parameters():
            g_grads.append(torch.mean(param.grad.view(-1)))
            break
        return rnn_grads, g_grads