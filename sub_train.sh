python sub_main8_raw.py \
--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
--train_batch_size 20 \
--sampler_num 20000 \
--store_folder ./output/raw \
--epochs 30
python sub_main7_raw_shu.py \
--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
--train_batch_size 20 \
--sampler_num 20000 \
--store_folder ./output/raw_shu \
--epochs 30

