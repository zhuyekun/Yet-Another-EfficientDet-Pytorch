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
import yaml
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
    force_category: Flag = False,
):
    coco = defaultdict(list)
    coco["images"], coco["categories"], coco["annotations"] = [], [], []
    coco["type"] = coco_type
    category_set = categories_set.copy()

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
                                len(category_set) + ctg_id_start, object_name
                            )
                            categories_id = len(category_set) + ctg_id_start
                            category_set[object_name] = categories_id
                            if not force_category:
                                coco["categories"].append(categories)
                            # print(category_set)
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
                        if force_category:
                            if categories_id in categories_set.values():
                                coco["annotations"].append(
                                    addAnnoItem(
                                        image["id"],
                                        categories_id,
                                        bndbox,
                                        len(coco["annotations"]),
                                    )
                                )
                        else:
                            coco["annotations"].append(
                                addAnnoItem(
                                    image["id"],
                                    categories_id,
                                    bndbox,
                                    len(coco["annotations"]),
                                )
                            )

    json.dump(coco, open(json_save_path, "w"))
    print("Total categories:{}".format(list(category_set.keys())))
    if force_category:
        print("forced categories:{}".format(list(categories_set.keys())))


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
                        elem.text = "DLLJ" + elem.text[4:]
                    if elem.text[0:2] == "坑槽":
                        elem.text = "KC" + elem.text[2:]
                    if elem.text[0:2] == "裂缝":
                        elem.text = "LF" + elem.text[2:]
                    if elem.text[0:2] == "积水":
                        elem.text = "JS" + elem.text[2:]
                    if elem.text[0:2] == "杂物":
                        elem.text = "ZW" + elem.text[2:]
                    if elem.text[0:2] == "线裂":
                        elem.text = "XL" + elem.text[2:]
                    if elem.text[0:2] == "补丁":
                        elem.text = "BD" + elem.text[2:]

                if elem.tag == "path":
                    path = PureWindowsPath(elem.text)
                    if path.name[0:4] == "道路垃圾":
                        p = Path(img_path)
                        c_path = p / ("DLLJ" + path.name[4:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "坑槽":
                        p = Path(img_path)
                        c_path = p / ("KC" + path.name[2:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "裂缝":
                        p = Path(img_path)
                        c_path = p / ("LF" + path.name[2:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "积水":
                        p = Path(img_path)
                        c_path = p / ("JS" + path.name[2:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "杂物":
                        p = Path(img_path)
                        c_path = p / ("ZW" + path.name[2:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "线裂":
                        p = Path(img_path)
                        c_path = p / ("XL" + path.name[2:])
                        elem.text = str(c_path)
                    if path.name[0:2] == "补丁":
                        p = Path(img_path)
                        c_path = p / ("BD" + path.name[2:])
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


def ctg_merge(xml_path, out_path, ctg_dict):
    path = Path(xml_path)
    for f in path.iterdir():
        if f.suffix == ".xml":
            tree = ET.parse(f)
            root = tree.getroot()
            for elem in root:
                if elem.tag == "object":
                    for subelem in elem:
                        if subelem.tag == "name":
                            for key in ctg_dict.keys():
                                if subelem.text in ctg_dict[key]:
                                    subelem.text = key
                                    break
            tree = ET.ElementTree(root)
            tree.write(out_path + "/" + f.name)


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
            # cv2.waitKey(0)


def load_yaml(yaml_path):
    infer_cfg = open(yaml_path)
    data = infer_cfg.read()
    yaml_reader = yaml.safe_load(data)

    return yaml_reader
