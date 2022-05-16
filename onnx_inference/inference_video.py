from pathlib import Path

from absl import app, flags

from utils import infer_video_onnx, load_onnx, load_yaml

flags.DEFINE_string("config_path", None, "Path to config file.")
flags.DEFINE_string("video_path", None, "Path to video.")
flags.DEFINE_string("output_path", None, "Path to output dir.")
flags.DEFINE_string(
    "exclusion_list", str([]), "Labels you don't want to appear in predictions"
)

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv
    flags.mark_flag_as_required("config_path")

    config = load_yaml(FLAGS.config_path)
    project_params = load_yaml(config["project_config"])

    obj_list = project_params["obj_list"]
    video_path = FLAGS.video_path
    exclusion_list = eval(FLAGS.exclusion_list)

    output_path = Path(FLAGS.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    ort_session = load_onnx(config["model_path"])

    infer_video_onnx(
        ort_session,
        config["threshold"],
        config["iou_threshold"],
        config["compound_coef"],
        obj_list,
        exclusion_list,
        video_path,
        str(output_path),
    )


if __name__ == "__main__":
    app.run(main)
