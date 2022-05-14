
export CUDA_VISIBLE_DEVICES="1"


# python train.py -c 2 -p untunnel-detection --batch_size 8 --lr 5e-3 --num_epochs 30 \
#  --load_weights weights/untunnel/2.5/efficientdet-d2_149_9000.pth --head_only True

python train.py -c 2 -p crack --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d2.pth --head_only True

python train.py -c 2 -p crack --batch_size 8 --lr 1e-4 --num_epochs 200 \
 --load_weights last --head_only False

python train.py -c 2 -p pothole --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d2.pth --head_only True

python train.py -c 2 -p pothole --batch_size 8 --lr 1e-4 --num_epochs 200 \
 --load_weights last --head_only False

 python train.py -c 2 -p pounding --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d2.pth --head_only True

python train.py -c 2 -p pounding --batch_size 8 --lr 1e-4 --num_epochs 200 \
 --load_weights last --head_only False