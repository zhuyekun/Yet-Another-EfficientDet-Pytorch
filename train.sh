# python train.py -c 4 -p waste_detection --batch_size 8 --lr 5e-3 --num_epochs 30 \
#  --load_weights weights/efficientdet-d4.pth --head_only True

# python train.py -c 4 -p waste_detection --batch_size 8 --lr 1e-4 --num_epochs 500 \
#  --load_weights last --head_only False

# python train.py -c 2 -p waste_detection --batch_size 8 --lr 1e-4 --num_epochs 500 \
#  --load_weights 'weights/waste-detection/efficientdet-d2_90_resized_anchor.pth' --head_only False

# python train.py -c 3 -p waste_detection --batch_size 8 --lr 1e-4 --num_epochs 500 \
#  --load_weights 'weights/waste-detection/efficientdet-d3_90_1000.pth' --head_only False

python train.py -c 0 -p detection --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d0.pth --head_only True

python train.py -c 0 -p detection --batch_size 8 --lr 1e-4 --num_epochs 300 \
 --load_weights last --head_only False