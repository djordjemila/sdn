# SDN
python3 train.py --task disentanglement --gpus 1 --exp_name SDN-3DShapes_  --make_checkpoint --batch 128 --batch_val 1024 --lrate 0.001 --lrate_decay 0.999 --sdn_max_scale 32 --sdn_min_scale 32 --sdn_nfeat_0 200 --sdn_num_dirs 1 --post_model IsoGaussian --prior_model IsoGaussian --obs_model DL --mix_components 5 --z_size 10 --h_size 32 --sampling_temperature 0.85 --amp --dataset 3DShapes --num_workers 2 --random_seed 1234 --check_val_every_n_epoch 1  --ds_list 5 10
# CNN baseline
#python3 train.py --task disentanglement --gpus 1 --exp_name SDN-3DShapes_  --make_checkpoint --batch 128 --batch_val 1024 --lrate 0.001 --lrate_decay 0.999 --sdn_max_scale 0 --sdn_min_scale 0 --sdn_nfeat_0 200 --sdn_num_dirs 1 --post_model IsoGaussian --prior_model IsoGaussian --obs_model DL --mix_components 5 --z_size 10 --h_size 32 --sampling_temperature 0.85 --amp --dataset 3DShapes --num_workers 2 --random_seed 1234 --check_val_every_n_epoch 1  --ds_list 5 10

