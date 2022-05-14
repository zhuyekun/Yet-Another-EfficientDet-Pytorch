from collections import defaultdict
from dataclasses import dataclass
from enum import Flag
from pathlib import Path, PurePath, PureWindowsPath

import cv2
import numpy as np
import onnxruntime
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (
    STANDARD_COLORS,
    get_index_label,
    invert_affine,
    plot_one_box,
    postprocess,
    preprocess,
    standard_to_bgr,
)


@dataclass
class model_params:
    compound_coef: int
    obj_list: list
    ratios: list
    scales: list
    model_path: str


def model_load(params: model_params):

    model = EfficientDetBackbone(
        compound_coef=params.compound_coef,
        num_classes=len(params.obj_list),
        # replace this part with your project's anchor config
        ratios=params.ratios,
        scales=params.scales,
    )

    model.load_state_dict(torch.load(params.model_path))
    model.requires_grad_(False)
    model.eval()

    return model


def model_eval(
    model,
    img_path,
    threshold,
    iou_threshold,
    use_cuda=True,
    use_float16=False,
    force_input_size=None,
    input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
):
    cudnn.fastest = True
    cudnn.benchmark = True
    input_size = (
        input_sizes[model.compound_coef]
        if force_input_size is None
        else force_input_size
    )
    if use_cuda:
        _model = model.cuda()
    if use_float16:
        _model = model.half()

    ori_imgs, framed_imgs, framed_metas = preprocess(
        img_path,
        max_size=input_size,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = _model(x)

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
    return out, ori_imgs


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


def visualization_pre(
    params: model_params,
    img_path,
    threshold,
    iou_threshold,
    use_cuda=True,
    use_float16=False,
    force_input_size=None,
    input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
    save_img: Flag = False,
    save_path: str = ".no/such/path",
):
    model = model_load(params)
    out, ori_imgs = model_eval(
        model,
        img_path,
        threshold,
        iou_threshold,
        use_cuda,
        use_float16,
        force_input_size,
        input_sizes,
    )

    display(out, ori_imgs, params.obj_list)

    if save_img and Path(save_path).exists():
        image_name = Path(img_path).name
        plt.savefig(save_path + "/" + image_name)


def load_onnx(path):
    if torch.cuda.is_available():
        ort_session = onnxruntime.InferenceSession(
            path, None, providers=["CUDAExecutionProvider"]
        )
    else:
        ort_session = onnxruntime.InferenceSession(path, None)
    print(torch.cuda.is_available())
    # ort_session = onnxruntime.InferenceSession(path)
    return ort_session


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def eval_onnx(
    ort_session,
    compound_coef,
    img_path,
    threshold,
    iou_threshold,
    use_float16=False,
    input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
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

    return out, ori_imgs


def display_bbox(
    out,
    ori_imgs,
    obj_list,
    save_img: Flag = False,
    save_path: str = ".no/such/path",
):

    display(out, ori_imgs, obj_list)

    if save_img and Path(save_path).exists():
        plt.savefig(save_path)


def visualization_bbox(
    img_name,
    json_path,
    img_path,
    obj_list,
    save_img=False,
    save_path: str = ".no/such/path",
):
    coco = COCO(json_path)
    # color_list = standard_to_bgr(STANDARD_COLORS)
    fig = plt.figure()

    list_imgIds = coco.getImgIds()
    for i in list_imgIds:
        img = coco.loadImgs(list_imgIds[i])[0]
        if img["file_name"] == img_name:
            image = cv2.imread(img_path + "/" + img["file_name"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image_name = img["file_name"]
            image_id = img["id"]

            img_annIds = coco.getAnnIds(imgIds=image_id)
            img_anns = coco.loadAnns(img_annIds)
            for j in range(len(img_anns)):
                x, y, w, h = img_anns[j]["bbox"]
                plot_one_box(
                    image,
                    [x, y, x + w, y + h],
                    label=obj_list[img_anns[j]["category_id"] - 1],
                    color=(250, 0, 0),
                )

            EPS = 1e-2
            width, height = image.shape[1], image.shape[0]
            dpi = fig.get_dpi()
            fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.axis("off")
            plt.imshow(image)
            if save_img and Path(save_path).exists():
                plt.savefig(save_path + "/" + img["file_name"])
            else:
                plt.show()
            return


def visualization_bbox_dir(
    json_path,
    img_path,
    obj_list,
    save_path: str = ".no/such/path",
):
    coco = COCO(json_path)
    fig = plt.figure()
    list_imgIds = coco.getImgIds()
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)
    for i in list_imgIds:
        img = coco.loadImgs(list_imgIds[i])[0]
        image = cv2.imread(img_path + "/" + img["file_name"])
        image_id = img["id"]

        img_annIds = coco.getAnnIds(imgIds=image_id)
        img_anns = coco.loadAnns(img_annIds)

        for j in range(len(img_anns)):
            x, y, w, h = img_anns[j]["bbox"]
            plot_one_box(
                image,
                [x, y, x + w, y + h],
                label=obj_list[img_anns[j]["category_id"] - 1],
                color=(250, 0, 0),
            )

        EPS = 1e-2
        width, height = image.shape[1], image.shape[0]
        dpi = fig.get_dpi()
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis("off")
        plt.imshow(image)
        # if Path(save_path).exists():
        path = save_path + "/" + img["file_name"]
        plt.savefig(path)


def visualization_pre_dir(
    params: model_params,
    img_dir_path,
    threshold,
    iou_threshold,
    use_cuda=True,
    use_float16=False,
    force_input_size=None,
    input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536],
    save_path: str = ".no/such/path",
):

    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)

    model = model_load(params)
    for img_path in Path(img_dir_path).iterdir():
        out, ori_imgs = model_eval(
            model,
            str(img_path),
            threshold,
            iou_threshold,
            use_cuda,
            use_float16,
            force_input_size,
            input_sizes,
        )
        path = save_path + "/" + img_path.name
        display(out, ori_imgs, params.obj_list, save_mode=True, save_path=path)
