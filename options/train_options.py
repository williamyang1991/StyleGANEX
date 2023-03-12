from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use') 
        self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        
        # new options for StyleGANEX
        self.parser.add_argument('--feat_ind', default=0, type=int, help='Layer index of G to accept the first-layer feature')
        self.parser.add_argument('--max_pooling', action="store_true", help='Apply max pooling or average pooling')
        self.parser.add_argument('--use_skip', action="store_true", help='Using skip connection from the encoder to the styleconv layers of G')
        self.parser.add_argument('--use_skip_torgb', action="store_true", help='Using skip connection from the encoder to the toRGB layers of G.')
        self.parser.add_argument('--skip_max_layer', default=7, type=int, help='Layer used for skip connection. 1,2,3,4,5,6,7 correspond to 4,8,16,32,64,128,256')
        self.parser.add_argument('--crop_face', action="store_true", help='Use aligned cropped face to predict style latent code w+')
        self.parser.add_argument('--affine_augment', action="store_true", help='Apply random affine transformation during training')
        self.parser.add_argument('--random_crop', action="store_true", help='Apply random crop during training')
        # for SR
        self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')
        self.parser.add_argument('--blind_sr', action="store_true", help='Whether training blind SR (will use ./datasetsffhq_degradation_dataset.py)')  
        # for sketch/mask to face translation
        self.parser.add_argument('--use_latent_mask', action="store_true", help='For segmentation/sketch to face translation, fuse w+ from two sources')
        self.parser.add_argument('--latent_mask', type=str, default='8,9,10,11,12,13,14,15,16,17', help='Comma-separated list of latents to perform style-mixing with')
        self.parser.add_argument('--res_num', default=2, type=int, help='Layer number of the resblocks of the translation network T')        
        # for video face toonify
        self.parser.add_argument('--toonify_weights', default=None, type=str, help='Path to Toonify StyleGAN model weights')
        # for video face editing
        self.parser.add_argument('--generate_training_data', action="store_true", help='Whether generating training data (for video editing) or load real data')
        self.parser.add_argument('--use_att', default=0, type=int, help='Layer of MLP used for attention, 0 not use attention')
        self.parser.add_argument('--editing_w_path', type=str, default=None, help='Path to the editing vector v')
        self.parser.add_argument('--zero_noise', action="store_true", help='Whether using zero noises')
        self.parser.add_argument('--direction_path', type=str, default=None, help='Path to the direction vector to augment generated data')
        
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
        self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
        self.parser.add_argument('--adv_lambda', default=0, type=float, help='Adversarial loss multiplier factor')
        self.parser.add_argument('--d_reg_every', default=16, type=int, help='Interval of the applying r1 regularization')
        self.parser.add_argument('--r1', default=1, type=float, help="weight of the r1 regularization")
        self.parser.add_argument('--tmp_lambda', default=0, type=float, help='Temporal loss multiplier factor')
        
        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

        # arguments for weights & biases support
        self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
