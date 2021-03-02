wget https://openaipublic.azureedge.net/glow-demo/data/imagenet-oord-tfr.tar
tar -xvf imagenet-oord-tfr.tar
python3 convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=mnt/host/imagenet-oord-tfr --lmdb_path=. --split=train
python3 convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=mnt/host/imagenet-oord-tfr --lmdb_path=. --split=validation
