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

#python sub_main93_MSPA_decoder.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/MSPA_decoder \
#--epochs 25
#还没跑93，先验证fusion

#python sub_main94_MSPA_Lightfusion.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/MSPA_Lightfusion \
#--epochs 25

#python sub_main97_MSPA_before_decoder.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/MSPA_before_decoder \
#--epochs 25

#python sub_main98_94_93.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/98 \
#--epochs 25

#python sub_main99_92_93_94.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/99 \
#--epochs 25

#python sub_main990_92_93_94_97.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 8 \
#--sampler_num 20000 \
#--store_folder ./output/990 \
#--epochs 25

#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 48 \
#--sampler_num 20000 \
#--store_folder ./output/001 \
#--epochs 25 \
#--num_workers 8

#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 32 \
#--sampler_num 20000 \
#--store_folder ./output/001 \
#--epochs 35 \
#--num_workers 8 \
#--resume /home/share3/zc/file/New_Net/output/001/checkpoints/epoch-13-ckpt.pt

#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 64 \
#--sampler_num 20000 \
#--store_folder ./output/001_crop320 \
#--epochs 25 \
#--num_workers 12

#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 64 \
#--sampler_num 20000 \
#--store_folder ./output/001_crop320_MSPA_up_before \
#--epochs 35 \
#--num_workers 12

#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 64 \
#--sampler_num 30000 \
#--store_folder ./output/001_crop320_MSPA_up_before_loss2 \
#--epochs 35 \
#--num_workers 12 \
#--loss_method AW

#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 64 \
#--sampler_num 30000 \
#--store_folder ./output/001_crop320_MSPA_up_before_res \
#--epochs 35 \
#--num_workers 12 \
#--loss_method HFL
#
#python sub_main001.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 64 \
#--sampler_num 30000 \
#--store_folder ./output/001_crop320_MSPA_up_before_res_loss \
#--epochs 35 \
#--num_workers 12 \
#--loss_method AW

#python sub_main_mix01.py \
#--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
#--train_batch_size 128 \
#--sampler_num 20000 \
#--store_folder ./output/PDDP_crop320 \
#--epochs 35 \
#--num_workers 12 \
#--loss_method HFL \
#--print_freq 250 \
#--lr_stepsize 10 \
#--learning_rate 8e-4

python sub_main_mix03.py \
--dataset /home/share/liuchangsong/edge_data/BSDS500_flip_rotate_pad/ \
--train_batch_size 64 \
--sampler_num 20000 \
--store_folder ./output/PDCNet_base \
--epochs 35 \
--num_workers 12 \
--loss_method HFL \
--print_freq 100 \
--lr_stepsize 3 \
--learning_rate 5e-4