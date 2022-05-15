
export CUDA_VISIBLE_DEVICES="0"

python inference.py \
--config_path=config/config.yml \
--img_path=examples/fig \
--output_path=output/fig


# python inference_video.py \
# --config_path=config/config_video.yml \
# --video_path=examples/videos/ch01_20220214162154.mp4 \
# --output_path=output/video \
# --exclusion_list='["pounding", "spiledmaterial"]'
