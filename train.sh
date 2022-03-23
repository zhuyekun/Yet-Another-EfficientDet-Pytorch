python train.py -c 4 -p waste_detection --batch_size 8 --lr 5e-3 --num_epochs 30 \
 --load_weights weights/efficientdet-d4.pth --head_only True

python train.py -c 4 -p waste_detection --batch_size 8 --lr 1e-4 --num_epochs 500 \
 --load_weights last --head_only False

python train.py -c 2 -p waste_detection --batch_size 8 --lr 1e-4 --num_epochs 500 \
 --load_weights 'weights/waste-detection/efficientdet-d2_90_resized_anchor.pth' --head_only False

python train.py -c 3 -p waste_detection --batch_size 8 --lr 1e-4 --num_epochs 500 \
 --load_weights 'weights/waste-detection/efficientdet-d3_90_1000.pth' --head_only False

 $USER_AT_HOST="matjxt-mz@login.hpc.sjtu.edu.cn"
$PUBKEYPATH="C:\Users\99225\.ssh\id_rsa.pub"

$pubKey=(Get-Content "$PUBKEYPATH" | Out-String); ssh "$USER_AT_HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '${pubKey}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"