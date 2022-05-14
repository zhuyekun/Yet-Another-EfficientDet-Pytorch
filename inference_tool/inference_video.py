from pathlib import Path

import cv2
from absl import app, flags
from utils_inference import infer_video_onnx, load_onnx, load_yaml

flags.DEFINE_string("config_path", None, "Path to config file.")
# flags.DEFINE_string("output_path", None, "Path to output.")

FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required("config_path")
    config = load_yaml(FLAGS.config_path)
    project_params = load_yaml(config["project_config"])

    obj_list = project_params["obj_list"]
    video_path = config["video_path"]
    exclusion_list = config["exclusion_list"]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[config["compound_coef"]]

    output_path = Path(config["output_path"])
    output_path.mkdir(exist_ok=True, parents=True)

    ort_session = load_onnx(config["model_path"])

    infer_video_onnx(
        ort_session,
        config["threshold"],
        config["iou_threshold"],
        input_size,
        obj_list,
        exclusion_list,
        video_path,
        str(output_path),
    )


if __name__ == "__main__":
    app.run(main)
