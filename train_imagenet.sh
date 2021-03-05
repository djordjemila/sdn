# SDN
bsub -n 16 -R 'rusage[ngpus_excl_p=8,mem=150000]' -R 'select[gpu_mtotal0>=30000]' -W 240 'python train.py --gpus 8 --exp_name SDN_ --make_checkpoint  --batch 32 --batch_val 1024 --depth 16 --ds_list 3 7 11 --lrate 0.002 --lrate_decay 1 --sdn_max_scale 32 --sdn_min_scale 0 --sdn_nfeat_0 424 --sdn_nfeat_diff 0 --sdn_num_dirs 3 --post_model IAF --prior_model IsoGaussian --obs_model DML --mix_components 10 --free_bits 0.01 --ema_coef 0.9995 --z_size 4 --h_size 200 --sampling_temperature 1.0 --amp --dataset ImageNet32 --distributed_backend ddp --num_workers 16 --random_seed 13 --check_val_every_n_epoch 1 --root ./data/'
# CNN baseline
# bsub -n 16 -R 'rusage[ngpus_excl_p=8,mem=150000]' -R 'select[gpu_mtotal0>=30000]' -W 240 'python train.py --gpus 8 --exp_name SDN_ --make_checkpoint  --batch 32 --batch_val 1024 --depth 16 --ds_list 3 7 11 --lrate 0.002 --lrate_decay 1 --sdn_max_scale 0 --sdn_min_scale 0 --sdn_nfeat_0 424 --sdn_nfeat_diff 0 --sdn_num_dirs 3 --post_model IAF --prior_model IsoGaussian --obs_model DML --mix_components 10 --free_bits 0.01 --ema_coef 0.9995 --z_size 4 --h_size 200 --sampling_temperature 1.0 --amp --dataset ImageNet32 --distributed_backend ddp --num_workers 16 --random_seed 13 --check_val_every_n_epoch 1 --root ./data/'