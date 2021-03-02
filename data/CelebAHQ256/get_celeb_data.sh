wget https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
tar -xvf celeba-tfr.tar
python3 convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=celeba-tfr --lmdb_path=. --split=train
python3 convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=celeba-tfr --lmdb_path=. --split=validation
