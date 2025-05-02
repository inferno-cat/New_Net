python sub_main_base.py \
--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
--train_batch_size 8 \
--sampler_num 20 \
--store_folder ./output/base \
--epochs 10
python sub_main_base.py \
--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
--train_batch_size 8 \
--sampler_num 20 \
--store_folder ./output/base \
--epochs 25 \
--resume ./output/base/checkpoints/epoch-10-ckpt.pt


