# Core Author: Zylo117
# Script's Author: winter2897

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from utils.eff_utils import load_yaml
from torch.backends import cudnn
from matplotlib import pyplot as plt
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from utils.vis_utils import model_params
import os

from utils.utils import (
    plot_one_box,
    STANDARD_COLORS,
    standard_to_bgr,
    get_index_label,
)

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default=None, help="/path/to/configs")
ap.add_argument("-v", "--video_path", type=str, default=None, help="/path/to/video")
ap.add_argument("--device", type=str, default="0")

args = ap.parse_args()

# print(args.device, args.compound_coef, args.weights, args.video_path)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# Video's path
video_src = args.video_path  # set int to use webcam, set str to read from a video file

compound_coef = args.compound_coef
# force_input_size = 1024
force_input_size = None  # set None to use default size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

params = []
for i, yaml_path in enumerate(args.project_config):
    project_params = load_yaml(str(yaml_path))
    anchor_ratios = eval(project_params["anchors_ratios"])
    anchor_scales = eval(project_params["anchors_scales"])
    obj_list = project_params["obj_list"]
    params.append(
        model_params(
            compound_coef, obj_list, anchor_ratios, anchor_scales, args.weights[i]
        )
    )

# exclusion_list = ["pounding"]
# color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = (
    input_sizes[compound_coef] if force_input_size is None else force_input_size
)

# load model
model = EfficientDetBackbone(
    compound_coef=compound_coef,
    num_classes=len(obj_list),
    ratios=anchor_ratios,
    scales=anchor_scales,
)
model.load_state_dict(torch.load(args.weights))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

video_path = Path(video_src)
videoname = str(video_path.parents[1] / video_path.stem) + "_output.avi"

# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# writer = cv2.VideoWriter(videoname, fourcc, 20.0, (1080, 720), True)

cap = cv2.VideoCapture(video_src)


video_frame_cnt = int(cap.get(7))
video_width = int(cap.get(3))
video_height = int(cap.get(4))
video_fps = int(cap.get(5))
# print(video_src.type)
video_path = Path(video_src)
videoname = str(video_path.parents[1] / video_path.stem) + "_output.avi"

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(
    videoname, fourcc, video_fps, (video_width, video_height), True
)

# function for display
def display(preds, imgs):
    # color_list = standard_to_bgr(STANDARD_COLORS)
    plt.rcParams["figure.figsize"] = (12.8, 7.2)
    for i in range(len(imgs)):
        if len(preds[i]["rois"]) == 0:
            return imgs[i]

        for j in range(len(preds[i]["rois"])):
            (x1, y1, x2, y2) = preds[i]["rois"][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]["class_ids"][j]]
            score = float(preds[i]["scores"][j])

            cv2.putText(
                imgs[i],
                "{}, {:.3f}".format(obj, score),
                (x1, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        return imgs[i]


def display_v2(preds, imgs, obj_list, exclusion_list=[]):
    color_list = standard_to_bgr(STANDARD_COLORS)
    plt.rcParams["figure.figsize"] = (12.8, 7.2)
    for i in range(len(imgs)):
        if len(preds[i]["rois"]) == 0:
            return imgs[i]

        for j in range(len(preds[i]["rois"])):
            (x1, y1, x2, y2) = preds[i]["rois"][j].astype(np.int)
            obj = obj_list[preds[i]["class_ids"][j]]
            score = float(preds[i]["scores"][j])

            if obj not in exclusion_list:
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                plot_one_box(
                    imgs[i],
                    [x1, y1, x2, y2],
                    label=obj,
                    score=score,
                    color=color_list[get_index_label(obj, obj_list)],
                )

        return imgs[i]


# Box
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

ind = 0
while True:
    ret, frame = cap.read()
    ind += 1
    if not ret:
        break

    # frame preprocessing
    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # model predict
    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(
            x,
            anchors,
            regression,
            classification,
            regressBoxes,
            clipBoxes,
            threshold,
            iou_threshold,
        )

    # result
    out = invert_affine(framed_metas, out)

    img_show = display_v2(out, ori_imgs, obj_list, exclusion_list=exclusion_list)
    # print("img_show:", len(img_show))
    # print("frame1:", len(ori_imgs[0]))
    # print("frame2:", len(frame[0]))
    # show frame by frame
    # cv2.imshow("frame", img_show)
    # cv2.imwrite("datasets/detection/video/first.png", img_show)
    writer.write(img_show)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    print("current_frame/all_frame:{}/{}".format(ind, video_frame_cnt))

cap.release()
writer.release()
cv2.destroyAllWindows()
