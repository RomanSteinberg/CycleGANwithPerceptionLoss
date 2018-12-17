from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', type=str, default='no',
                                 help='options to continue training. [no | load_G_and_D | load_G ]')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--smoothed_label', type=str, default='no', 
                                 help='use smoothed labels for D net. [no, | fixed | random]')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_feat_AfB', type=float, default=0,
                                 help='weight for perception loss between real A and fake B ')
        self.parser.add_argument('--lambda_feat_BfA', type=float, default=0,
                                 help='weight for perception loss between real B and fake A ')
        self.parser.add_argument('--lambda_feat_fArecB', type=float, default=0,
                                 help='weight for perception loss between fake A and reconstructed B ')
        self.parser.add_argument('--lambda_feat_fBrecA', type=float, default=0,
                                 help='weight for perception loss between fake B and reconstructed A ')
        self.parser.add_argument('--lambda_feat_ArecA', type=float, default=0,
                                 help='weight for perception loss between real A and reconstructed A ')
        self.parser.add_argument('--lambda_feat_BrecB', type=float, default=0,
                                 help='weight for perception loss between real B and reconstruced B ')
        self.parser.add_argument('--lambda_feat_color', type=float, default=0,
                                 help='weight for color loss between real A and fake B')
        self.parser.add_argument('--lambda_D', type=float, default=1.0,
                                 help='weight for discriminators loss')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--identity_loss', type=str, default='l1', help='l1, lab_mse, lab_mae')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/['
                                      'opt.name]/web/')
        self.parser.add_argument('--data_description', type=str, default='', help='data description if necessary')
        self.isTrain = True
