#
#python sub_main_base.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/base \
#--epochs 25

#python sub_main_base.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20 \
#--store_folder ./output/base \
#--epochs 25 \
#--resume ./output/base/checkpoints/epoch-10-ckpt.pt

#python sub_main2_MSPA.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/MSPA \
#--epochs 25

#python sub_main92_MSPA_up.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/MSPA_up \
#--epochs 25

python sub_main932_MSPA_decoder.py \
--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
--train_batch_size 8 \
--sampler_num 20000 \
--store_folder ./output/MSPA_decoder \
--epochs 25

