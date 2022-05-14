
export CUDA_VISIBLE_DEVICES="1"


# mv projects/0414split_2.8.yml projects/0414split.yml

python train.py -c 2 -p 0509split --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d2.pth --head_only True

python train.py -c 2 -p 0509split --batch_size 8 --lr 1e-4 --num_epochs 300 \
 --load_weights last --head_only False

# # mv projects/0414split.yml projects/0414split_2.8.yml
# # mv projects/0422split.yml projects/0422split_d2.yml
mv logs/0509split logs/0509split_d2

python train.py -c 4 -p 0509split --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d4.pth --head_only True

python train.py -c 4 -p 0509split --batch_size 8 --lr 1e-4 --num_epochs 300 \
 --load_weights last --head_only False

# mv projects/0414split.yml projects/0414split_2.8.yml
# mv projects/0422split.yml projects/0422split_d4.yml
mv logs/0509split logs/0509split_d4
# mv logs/0414split logs/0414split_2.8

# python train.py -c 2 -p 0414split --batch_size 8 --lr 5e-3 --num_epochs 30 \
#  --load_weights weights/efficientdet-d2.pth --head_only True

# python train.py -c 2 -p 0414split --batch_size 8 --lr 1e-4 --num_epochs 300 \
#  --load_weights last --head_only False


# mv projects/0414split.yml projects/0414split_2.5.yml
# mv projects/0414split_3.yml projects/0414split.yml

# mv logs/0414split logs/0414split_2.5

# python train.py -c 2 -p 0414split --batch_size 8 --lr 5e-3 --num_epochs 30 \
#  --load_weights weights/efficientdet-d2.pth --head_only True

# python train.py -c 2 -p 0414split --batch_size 8 --lr 1e-4 --num_epochs 300 \
#  --load_weights last --head_only False


# mv projects/0414split.yml projects/0414split_3.yml
# mv logs/0414split logs/0414split_3