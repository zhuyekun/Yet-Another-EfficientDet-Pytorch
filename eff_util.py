from enum import Flag
import json
import xml.etree.ElementTree as ET
import cv2
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from collections import defaultdict
from pathlib import Path, PureWindowsPath, PurePath
import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes

from utils.utils import (
    preprocess,
    invert_affine,
    postprocess,
    plot_one_box,
    STANDARD_COLORS,
    standard_to_bgr,
    get_index_label,
)


def addAnnoItem(image_id, category_id, bbox, annotation_id):
    annotation_item = dict()
    annotation_item["segmentation"] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item["segmentation"].append(seg)

    annotation_item["area"] = bbox[2] * bbox[3]
    annotation_item["iscrowd"] = 0
    annotation_item["ignore"] = 0
    annotation_item["image_id"] = image_id
    annotation_item["bbox"] = bbox
    annotation_item["category_id"] = category_id
    annotation_id += 1
    annotation_item["id"] = annotation_id

    return annotation_item


def addCtgItem(category_id, object_name):
    categories = dict()
    categories["supercategory"] = "none"
    categories_id = category_id
    categories["id"] = categories_id
    categories["name"] = object_name
    return categories


def parseXmlFiles(
    xml_path: str,
    json_save_path: str,
    coco_type: str = "instances",
    ctg_id_start: int = 1,
    categories_set: dict = {},
):
    coco = defaultdict(list)
    coco["images"], coco["categories"], coco["annotations"] = [], [], []
    coco["type"] = coco_type
    category_set = categories_set

    if len(category_set) != 0:
        for key, item in category_set.items():
            categories = addCtgItem(item, key)
            coco["categories"].append(categories)

    path = Path(xml_path)

    for f in path.iterdir():
        if f.suffix != ".xml":
            continue

        tree = ET.parse(f)
        root = tree.getroot()
        if root.tag != "annotation":
            raise Exception(
                "pascal voc xml root element should be annotation, rather than {}".format(
                    root.tag
                )
            )
        image = dict()
        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            if elem.tag == "folder":
                continue

            if elem.tag == "filename":

                file_name = elem.text
                image["file_name"] = file_name
                image["id"] = len(coco["images"])

            if elem.tag == "size":
                for subelem in elem:
                    if subelem.tag == "width" or subelem.tag == "height":
                        image[subelem.tag] = int(subelem.text)

            if len(image) == 4 and len(coco["images"]) == image["id"]:
                coco["images"].append(image)
                print("add image with {}".format(file_name))

            if elem.tag == "object":
                bbox = dict()
                # categories_id = None
                for subelem in elem:
                    if subelem.tag == "name":
                        object_name = subelem.text
                        if object_name not in category_set:
                            categories = addCtgItem(
                                len(coco["categories"]) + ctg_id_start, object_name
                            )

                            category_set[object_name] = categories_id
                            coco["categories"].append(categories)
                            # print(coco["categories"])
                        else:
                            categories_id = category_set[object_name]
                    if subelem.tag == "bndbox":
                        for e in subelem:
                            bbox[e.tag] = int(e.text)

                    if len(bbox) > 0:
                        bndbox = []
                        # x
                        bndbox.append(bbox["xmin"])
                        # y
                        bndbox.append(bbox["ymin"])
                        # w
                        bndbox.append(bbox["xmax"] - bbox["xmin"])
                        # h
                        bndbox.append(bbox["ymax"] - bbox["ymin"])

                        # print(categories, image)
                        # if categories_id in [1]:
                        coco["annotations"].append(
                            addAnnoItem(
                                image["id"],
                                categories_id,
                                bndbox,
                                len(coco["annotations"]),
                            )
                        )

    json.dump(coco, open(json_save_path, "w"))


# delete '道路垃圾' in xml and correct dataset path
def preprocessXml(xml_path, img_path):
    path = Path(xml_path)
    for f in path.iterdir():
        if f.suffix == ".xml":
            tree = ET.parse(f)
            root = tree.getroot()
            for elem in root:
                if elem.tag == "filename":
                    if elem.text[0:4] == "道路垃圾":
                        elem.text = "sp" + elem.text[4:]
                    if elem.text[0:2] == "坑槽":
                        elem.text = "pot" + elem.text[2:]
                    if elem.text[0:2] == "裂缝":
                        elem.text = "cr" + elem.text[2:]

                if elem.tag == "path":
                    path = PureWindowsPath(elem.text)
                    if path.name[0:4] == "道路垃圾":
                        p = Path(img_path)
                        c_path = p / ("sp" + path.name[4:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "坑槽":
                        p = Path(img_path)
                        c_path = p / ("pot" + path.name[2:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "裂缝":
                        p = Path(img_path)
                        c_path = p / ("cr" + path.name[2:])
                        elem.text = str(c_path)
            tree = ET.ElementTree(root)
            tree.write(f)


def correct_path(xml_path, img_path):
    path = Path(xml_path)
    for f in path.iterdir():
        if f.suffix == ".xml":
            tree = ET.parse(f)
            root = tree.getroot()
            for elem in root:
                if elem.tag == "path":
                    p = Path(elem.text)
                    c_path = Path(img_path) / p.name
                    elem.text = str(c_path)
            tree = ET.ElementTree(root)
            tree.write(f)


def display(preds, imgs, obj_list, imshow=True):
    color_list = standard_to_bgr(STANDARD_COLORS)
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

        if imshow:
            cv2.imshow("img", imgs[i])
            cv2.waitKey(0)


def visualization_bbox(
    img_name,
    json_path,
    img_path,
    save_img: Flag = False,
    save_path: str = ".no/such/path",
):
    coco = COCO(json_path)

    list_imgIds = coco.getImgIds()  # 获取含有该给定类别的所有图片的id
    # print(list_imgIds)
    for i in list_imgIds:
        img = coco.loadImgs(list_imgIds[i])[0]
        # print(img["file_name"])
        if img["file_name"] == img_name:
            image = cv2.imread(img_path + "/" + img["file_name"])  # 读取图像
            image_name = img["file_name"]  # 读取图像名字
            image_id = img["id"]  # 读取图像id

            img_annIds = coco.getAnnIds(imgIds=image_id)
            img_anns = coco.loadAnns(img_annIds)

            for i in range(len(img_annIds)):
                x, y, w, h = img_anns[i]["bbox"]  # 读取边框
                image = cv2.rectangle(
                    image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2
                )

            plt.rcParams["figure.figsize"] = (12.8, 7.2)
            plt.imshow(image)
            if save_img and Path(save_path).exists():
                plt.savefig(save_path + "/" + img["file_name"])
            else:
                plt.show()
            return


def visualization_pre(
    compound_coef: int,
    img_path: str,
    threshold: float,
    iou_threshold: float,
    obj_list: list,
    ratios: list,
    scales: list,
    model_path: str,
    save_img: Flag = False,
    save_path: str = ".no/such/path",
):
    force_input_size = None  # set None to use default size

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = (
        input_sizes[compound_coef] if force_input_size is None else force_input_size
    )
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(
        compound_coef=compound_coef,
        num_classes=len(obj_list),
        # replace this part with your project's anchor config
        ratios=ratios,
        scales=scales,
    )
    # scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load(model_path))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        # print(classification)

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
    # print(out)
    # print(x.shape[0])
    out = invert_affine(framed_metas, out)

    # print(out)

    for i in range(len(ori_imgs)):
        if len(out[i]["rois"]) == 0:
            # print(len(ori_imgs))
            continue
        ori_imgs[i] = ori_imgs[i].copy()
        for j in range(len(out[i]["rois"])):
            (x1, y1, x2, y2) = out[i]["rois"][j].astype(np.int)
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[i]["class_ids"][j]]
            score = float(out[i]["scores"][j])

            cv2.putText(
                ori_imgs[i],
                "{}, {:.3f}".format(obj, score),
                (x1, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            plt.imshow(ori_imgs[i])
    if save_img and Path(save_path).exists():
        image_name = Path(img_path).name
        plt.savefig(save_path + "/" + image_name)


def visualization_bbox_dir(
    json_path,
    img_path,
    save_path: str = ".no/such/path",
):
    coco = COCO(json_path)

    list_imgIds = coco.getImgIds()  # 获取含有该给定类别的所有图片的id
    # print(list_imgIds)
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)
    for i in list_imgIds:
        img = coco.loadImgs(list_imgIds[i])[0]
        # print(img["file_name"])
        image = cv2.imread(img_path + "/" + img["file_name"])  # 读取图像
        image_name = img["file_name"]  # 读取图像名字
        image_id = img["id"]  # 读取图像id

        img_annIds = coco.getAnnIds(imgIds=image_id)
        img_anns = coco.loadAnns(img_annIds)

        for i in range(len(img_annIds)):
            # if img_anns[i]["filename"] == "132.jpg"
            x, y, w, h = img_anns[i]["bbox"]  # 读取边框
            image = cv2.rectangle(
                image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2
            )

        plt.rcParams["figure.figsize"] = (12.8, 7.2)
        plt.imshow(image)
        # if Path(save_path).exists():
        path = save_path + "/" + img["file_name"]
        plt.savefig(path)


def visualization_pre_dir(
    compound_coef: int,
    img_dir_path: str,
    threshold: float,
    iou_threshold: float,
    obj_list: list,
    ratios: list,
    scales: list,
    model_path: str,
    save_path: str = ".no/such/path",
):
    force_input_size = None  # set None to use default size

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = (
        input_sizes[compound_coef] if force_input_size is None else force_input_size
    )

    model = EfficientDetBackbone(
        compound_coef=compound_coef,
        num_classes=len(obj_list),
        # replace this part with your project's anchor config
        ratios=ratios,
        scales=scales,
    )
    # scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model.load_state_dict(torch.load(model_path))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    for img_path in Path(img_dir_path).iterdir():
        ori_imgs, framed_imgs, framed_metas = preprocess(
            str(img_path), max_size=input_size
        )

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(
            0, 3, 1, 2
        )

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            # print(classification)

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
        # print(out)
        # print(x.shape[0])
        out = invert_affine(framed_metas, out)

        # print(out)

        for i in range(len(ori_imgs)):
            if len(out[i]["rois"]) == 0:
                # print(len(ori_imgs))
                continue
            ori_imgs[i] = ori_imgs[i].copy()
            for j in range(len(out[i]["rois"])):
                (x1, y1, x2, y2) = out[i]["rois"][j].astype(np.int)
                cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = obj_list[out[i]["class_ids"][j]]
                score = float(out[i]["scores"][j])

                cv2.putText(
                    ori_imgs[i],
                    "{}, {:.3f}".format(obj, score),
                    (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
                plt.rcParams["figure.figsize"] = (12.8, 7.2)
                plt.imshow(ori_imgs[i])
        path = save_path + "/" + img_path.name
        plt.savefig(path)
