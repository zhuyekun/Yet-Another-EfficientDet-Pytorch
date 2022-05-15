import time
import warnings
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import webcolors
import yaml
from matplotlib import pyplot as plt
from torch.backends import cudnn
from torchvision.ops.boxes import batched_nms
from tqdm import tqdm


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.0
        xmin = x_centers - w / 2.0
        ymax = y_centers + h / 2.0
        xmax = x_centers + w / 2.0

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


def preprocess(
    *image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
):
    ori_imgs = [
        cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in image_path
    ]
    # ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [
        aspectaware_resize_padding(img, max_size, max_size, means=None)
        for img in normalized_imgs
    ]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return (
        canvas,
        new_w,
        new_h,
        old_w,
        old_h,
        padding_w,
        padding_h,
    )


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]["rois"]) == 0:
            continue
        else:
            if metas is float:
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / metas
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / (
                    new_w / old_w
                )
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / (
                    new_h / old_h
                )
    return preds


def postprocess(
    x,
    anchors,
    regression,
    classification,
    regressBoxes,
    clipBoxes,
    threshold,
    iou_threshold,
):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    # print(scores)
    # print(scores > threshold)
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            # print(scores_over_thresh[i].sum())
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(
            1, 0
        )
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(
            transformed_anchors_per,
            scores_per[:, 0],
            classes_,
            iou_threshold=iou_threshold,
        )

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append(
                {
                    "rois": boxes_.cpu().numpy(),
                    "class_ids": classes_.cpu().numpy(),
                    "scores": scores_.cpu().numpy(),
                }
            )
        else:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )

    return out


def preprocess_video(
    *frame_from_video,
    max_size=512,
    mean=(0.406, 0.456, 0.485),
    std=(0.225, 0.224, 0.229),
):
    ori_imgs = frame_from_video
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [
        aspectaware_resize_padding(img, max_size, max_size, means=None)
        for img in normalized_imgs
    ]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def load_yaml(yaml_path):
    infer_cfg = open(yaml_path)
    data = infer_cfg.read()
    yaml_reader = yaml.safe_load(data)

    return yaml_reader


def load_onnx(path):
    if torch.cuda.is_available():
        ort_session = onnxruntime.InferenceSession(
            path, None, providers=["CUDAExecutionProvider"]
        )
    else:
        ort_session = onnxruntime.InferenceSession(path, None)
    # print(torch.cuda.is_available())
    # ort_session = onnxruntime.InferenceSession(path)
    return ort_session


def eval_onnx(
    ort_session,
    compound_coef,
    img_path,
    threshold,
    iou_threshold,
    use_float16=False,
    input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
    print_fps=False,
):
    cudnn.fastest = True
    cudnn.benchmark = True
    input_size = input_sizes[compound_coef]

    ori_imgs, framed_imgs, framed_metas = preprocess(
        img_path,
        max_size=input_size,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    x = np.stack(framed_imgs, 0)
    x = np.moveaxis(x, [0, 3, 1, 2], [0, 1, 2, 3])

    ort_inputs = {ort_session.get_inputs()[0].name: x}
    if print_fps:
        print("model inferring and postprocessing...")
        t1 = time.time()
    # _, _, _, _, _, regression, classification, anchors = ort_session.run(
    #     None, ort_inputs
    # )
    regression, classification, anchors = [
        torch.from_numpy(i) for i in ort_session.run(None, ort_inputs)[-3:]
    ]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

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
    out = invert_affine(framed_metas, out)
    if print_fps:
        t2 = time.time()
        tact_time = t2 - t1
        print(f"{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1")

    return out, ori_imgs


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def display(preds, imgs, obj_list, save_mode=False, save_path=""):
    color_list = standard_to_bgr(STANDARD_COLORS)
    fig = plt.figure()
    EPS = 1e-2
    for i in range(len(imgs)):
        if len(preds[i]["rois"]) == 0:
            continue

        imgs[i] = imgs[i].copy()
        for j in range(len(preds[i]["rois"])):
            x1, y1, x2, y2 = preds[i]["rois"][j].astype(np.int)
            obj = obj_list[preds[i]["class_ids"][j]]
            score = float(preds[i]["scores"][j])
            plot_one_box(
                imgs[i],
                [x1, y1, x2, y2],
                label=obj,
                score=score,
                color=color_list[get_index_label(obj, obj_list)],
            )
            width, height = imgs[0].shape[1], imgs[0].shape[0]
            dpi = fig.get_dpi()
            fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.axis("off")
            # fig = plt.gcf()
            plt.imshow(imgs[i])
            if save_mode:
                plt.savefig(save_path)

        # if imshow:
        #     cv2.imshow("img", imgs[i])


def display_bbox(
    out,
    ori_imgs,
    obj_list,
    save_img=False,
    save_path: str = ".no/such/path",
):

    display(out, ori_imgs, obj_list)

    if save_img and Path(save_path).parent.exists():
        plt.savefig(save_path)


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


STANDARD_COLORS = [
    "LawnGreen",
    "Chartreuse",
    "Aqua",
    "Beige",
    "Azure",
    "BlanchedAlmond",
    "Bisque",
    "Aquamarine",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "AntiqueWhite",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Salmon",
    "Tan",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "AliceBlue",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "MediumAquaMarine",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Green",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "GreenYellow",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
]


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label and score:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(
            str("{:.0%}".format(score)), 0, fontScale=float(tl) / 3, thickness=tf
        )[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(
            img,
            "{}: {:.0%}".format(label, score),
            (c1[0], c1[1] - 2),
            0,
            float(tl) / 3,
            [0, 0, 0],
            thickness=tf,
            lineType=cv2.FONT_HERSHEY_SIMPLEX,
        )
    elif label and not score:
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(
            img,
            "{}".format(label),
            (c1[0], c1[1] - 2),
            0,
            float(tl) / 3,
            [0, 0, 0],
            thickness=tf,
            lineType=cv2.FONT_HERSHEY_SIMPLEX,
        )


def infer_video_onnx(
    ort_session,
    threshold,
    iou_threshold,
    input_size,
    obj_list,
    exclusion_list,
    video_src,
    output_path,
):
    cudnn.fastest = True
    cudnn.benchmark = True
    output_path = Path(output_path)
    video_path = Path(video_src)
    videoname = str(output_path / video_path.stem) + "_output.avi"
    cap = cv2.VideoCapture(video_src)

    video_frame_cnt = int(cap.get(7))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        videoname, fourcc, video_fps, (video_width, video_height), True
    )
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    pbar = tqdm(total=video_frame_cnt)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame preprocessing
        ori_imgs, framed_imgs, framed_metas = preprocess_video(
            frame, max_size=input_size
        )

        x = np.stack(framed_imgs, 0)
        x = np.moveaxis(x, [0, 3, 1, 2], [0, 1, 2, 3])

        ort_inputs = {ort_session.get_inputs()[0].name: x}
        _, _, _, _, _, regression, classification, anchors = ort_session.run(
            None, ort_inputs
        )
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(
            x,
            torch.from_numpy(anchors),
            torch.from_numpy(regression),
            torch.from_numpy(classification),
            regressBoxes,
            clipBoxes,
            threshold,
            iou_threshold,
        )
        out = invert_affine(framed_metas, out)

        img_show = display_v2(out, ori_imgs, obj_list, exclusion_list=exclusion_list)
        writer.write(img_show)
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
