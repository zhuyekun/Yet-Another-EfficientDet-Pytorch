# python efficientdet_test_videos.py -c 2 -w 'weights/untunnel/2.5/efficientdet-d2_99_6000.pth'\
#     -v 'datasets/detection/video/mov_clear/ch01_20220214142025.mp4' \
#     --device "1"

# python efficientdet_test_videos.py -c 2 -w 'weights/untunnel/2.5/efficientdet-d2_99_6000.pth'\
#     -v 'datasets/detection/video/mov_clear/ch01_20220214162321.mp4' \
#     --device "1"

python efficientdet_test_videos.py -c 2 -w 'logs/0422split_d2/efficientdet-d2_80_131000.pth'\
    -v 'datasets/0414split/video/04-24.mp4' \
    --device "0"

# python efficientdet_test_videos.py -c 2 -w 'weights/untunnel/2.5/efficientdet-d2_99_6000.pth'\
#     -v 'datasets/detection/video/mov_clear/ch01_20220214162545.mp4' \
#     --device "1"