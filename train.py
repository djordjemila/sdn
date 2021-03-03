import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
from lib.DensityVAE import DensityVAE
from lib.DisentanglementVAE_old import DisentanglementVAE
from lib.utils import run_cuda_diagnostics
import time
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lib.utils import count_pars
import torch.autograd.profiler as profiler
from pytorch_lightning.callbacks import LearningRateLogger


def main(arguments=None):

    # parse arguments passed via (i) python script; or (ii) command line;
    parser = get_parser()
    args = parser.parse_args(arguments.split()) if arguments else parser.parse_args()

    # at the moment, the following tasks are supported
    assert args.task in ['density_estimation', 'disentanglement']
    task_to_model_map = dict({'density_estimation': DensityVAE, 'disentanglement':    DisentanglementVAE})
    model_name = task_to_model_map[args.task]

    # for reproducibility and synchronization
    pl.seed_everything(args.random_seed)

    # cuda diagnostics
    run_cuda_diagnostics(requested_num_gpus=args.gpus)

    # set num workers to 0 in profile mode -- this is required due to pytorch profiler limitations
    if args.profiler_mode:
        args.num_workers = 0
        args.fast_dev_run = 1

    # instantiate model
    model = model_name(**vars(args))
    print("\nTraining diagnostics:")
    print("---------------------")
    print("model signature", model.signature)
    print("make checkpoints? ", args.make_checkpoint)
    print("use amp? ", args.amp)
    print("total number of parameters: ", count_pars(model))

    # construct full experiment name which contains hyper parameters and a time-stamp
    full_exp_name = args.exp_name + "_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%f')  \
                                  + "_" + model.signature

    # checkpoint directory
    checkpoint_dir = os.getcwd()+'/checkpoints/'+args.exp_name
    os.makedirs(checkpoint_dir, exist_ok=True)

    # load existing, or create a new checkpoint
    detected_checkpoint = None
    if args.use_checkpoint:
        checkpoint_list = os.listdir(checkpoint_dir)
        checkpoint_list.sort(reverse=True)
        for checkpoint in checkpoint_list:
            if checkpoint.startswith(model.signature):
                detected_checkpoint = checkpoint_dir + "/" + checkpoint
                full_exp_name = "CHK_" + full_exp_name
                print("Checkpoint found.")
                break

    # setup a checkpoint callback
    checkpoint_callback = None
    if args.make_checkpoint:
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir, monitor='val_loss',
                                              prefix=model.signature, period=args.check_val_every_n_epoch)

    # empty cache now that the model is created
    torch.cuda.empty_cache()

    # train the model
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=WandbLogger(name=args.exp_name+"_"+model.signature,
                                                               project='sdn-'+args.dataset),
                                            progress_bar_refresh_rate=500,
                                            row_log_interval=500,
                                            log_save_interval=500,
                                            distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
                                            terminate_on_nan=True,
                                            checkpoint_callback=checkpoint_callback,
                                            resume_from_checkpoint=detected_checkpoint,
                                            callbacks=[LearningRateLogger(logging_interval='step')],
                                            benchmark=True
                                            )

    print("\nCommencing training:")
    print("----------------------")
    if args.profiler_mode:
        with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
            with profiler.record_function("model_inference"):
                trainer.fit(model)
        print("sort_by_self_cuda_time_total")
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=25))
        print("sort_by_self_cuda_memory_usage")
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=25))
    else:
        trainer.fit(model)


def get_parser():

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # General settings
    parser.add_argument('--root', type=str, default="./data/", help='Path to data folder.')
    parser.add_argument('--random_seed', type=int, default=13, help='Random seed.')
    parser.add_argument('--exp_name', type=str, default="NoName", help='The name of the experiment.')
    parser.add_argument('--task', type=str, default="density_estimation",
                        help='Pick between density_estimation and disentanglement based on the task.')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Dataset name.')
    parser.add_argument('--iters', type=int, default=1000000, help='Maximum number of training iterations.')
    parser.add_argument('--amp', action='store_true', help='Apply automatic mixed precision (AMP).')
    parser.add_argument('--make_checkpoint', action='store_true', help='Flag to indicate whether to make a checkpoint.')
    parser.add_argument('--use_checkpoint', action='store_true', help='Flag to indicate whether to resume training.')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers.')
    parser.add_argument('--figsize', type=int, default=10, help='Size of images logged during training.')
    parser.add_argument('--sampling_temperature', type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument('--nbits', type=int, default=8, help='Number of bits per pixel, when quantization is done.')
    parser.add_argument('--evaluation_mode', action='store_true', help='If model is used only for evaluation.')
    parser.add_argument('--profiler_mode', action='store_true', help='Run only one training iteration to do profiling.')
    # VAE and training settings
    parser.add_argument('--z_size', type=int, default=20, help='Number of stochastic feature maps per layer.')
    parser.add_argument('--h_size', type=int, default=160, help='Number of deterministic feature maps per layer.')
    parser.add_argument('--batch', type=int, default=32, help='Batch size.')
    parser.add_argument('--batch_val', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--lrate', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--lrate_decay', type=float, default=1, help='Exponential learning rate decay. Def: no decay.')
    parser.add_argument('--free_bits', type=float, default=0, help='KL free-bits.')
    parser.add_argument('--depth', type=int, default=5, help='The depth of VAE i.e. number of ladder layers.')
    parser.add_argument('--post_model', type=str, default="IsoGaussian",
                        help='Posterior model. Supported are IsoGaussian or IAF at the moment.')
    parser.add_argument('--prior_model', type=str, default="IsoGaussian",
                        help='Prior model. Supported are IsoGaussian or IAF at the moment.')
    parser.add_argument('--obs_model', type=str, default="DL",
                        help='Observation model. Supported are DML or DL at the moment.')
    parser.add_argument('--mix_components', type=int, default=30, help='Number of mixture components, for DML.')
    parser.add_argument('--ema_coef', type=float, default=1, help='Exponential moving average (EMA) coefficient.')
    parser.add_argument('--beta_rate', type=float, default=1,
                        help='KL annealing rate: the per iteration increment. 1 means there is no KL annealing.')
    parser.add_argument('--ds_list', nargs='*', type=int,
                        help='Provided as a list of integers. Indicates at which layers the downsampling is performed.')
    parser.add_argument('--downsample_first', action='store_true', help='Downsample in the first layer of encoder.')
    # SDNLayer settings
    parser.add_argument('--sdn_max_scale', type=int, default=64, help='Maximum scale to apply SDNLayer on.')
    parser.add_argument('--sdn_min_scale', type=int, default=0, help='Minimum scale to apply SDNLayer on.')
    parser.add_argument('--sdn_nfeat_0', type=int, default=300, help='Number of features at largest-scale SDNLayer.')
    parser.add_argument('--sdn_nfeat_diff', type=int, default=0,
                        help='Each time we downscale, we reduce the number of SDNLayer features by this number.')
    parser.add_argument('--sdn_num_dirs', type=int, default=1, help='Number of SDNLayer directions.')

    return parser


if __name__ == '__main__':
    main()
