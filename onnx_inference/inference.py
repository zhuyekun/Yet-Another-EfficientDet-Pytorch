from pathlib import Path

from absl import app, flags

from utils import display_bbox, eval_onnx, load_onnx, load_yaml, read_imgs

flags.DEFINE_string("config_path", None, "Path to config file.")
flags.DEFINE_string("img_path", None, "Path to image or image dir.")
flags.DEFINE_string("output_path", None, "Path to output.")

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv
    flags.mark_flag_as_required("config_path")
    flags.mark_flag_as_required("img_path")
    flags.mark_flag_as_required("output_path")

    img_path = Path(FLAGS.img_path)
    output_path = Path(FLAGS.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    config = load_yaml(FLAGS.config_path)
    project_params = load_yaml(config["project_config"])
    obj_list = project_params["obj_list"]

    ort_session = load_onnx(config["model_path"])
    if img_path.is_dir():
        for _, img in enumerate(img_path.iterdir()):
            imgs = read_imgs(str(img))
            out, ori_imgs = eval_onnx(
                ort_session,
                config["compound_coef"],
                imgs,
                config["threshold"],
                config["iou_threshold"],
                input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
            )
            save_path = str(output_path / img.stem) + "_infer.jpg"
            display_bbox(out, ori_imgs, obj_list, save_img=True, save_path=save_path)
    else:
        imgs = read_imgs(str(img_path))
        out, ori_imgs = eval_onnx(
            ort_session,
            config["compound_coef"],
            imgs,
            config["threshold"],
            config["iou_threshold"],
            input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
        )

        save_path = str(output_path / img_path.stem) + "_infer.jpg"
        display_bbox(out, ori_imgs, obj_list, save_img=True, save_path=save_path)


if __name__ == "__main__":
    app.run(main)
