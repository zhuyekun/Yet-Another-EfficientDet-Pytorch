import torch
import torch.onnx
from torch import nn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.eff_utils import load_yaml

# from utils.utils import preprocess

COMPOUND_COEF = 2
BATCH_SIZE = 1
INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
IMG_SIZE = INPUT_SIZES[COMPOUND_COEF]

CONFIG_PATH = "onnx_inference/model/v1/0509split.yml"
CKPT_PATH = f"logs/weights/efficientdet-d{COMPOUND_COEF}_125_253500.pth"


def decode_anchors_to_centersize(anchors):
    """Transforms anchor boxes' encoding from box-corner to center-size.
    Box-corner encoding is of form: {ymin, ymax, xmin, xmax}
    Center-size encoding is of form: {y_center, x_center, height, width}
    This is used for TFLite's custom NMS post-processing.
    Args:
        boxes: predicted box regression targets.
        anchors: anchors on all feature levels.
    Returns:
        outputs: anchor_boxes in center-size encoding.
    """
    ycenter_a = (anchors[..., 0] + anchors[..., 2]) / 2
    xcenter_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    return torch.stack([ycenter_a, xcenter_a, ha, wa], dim=-1)


class EfficientDetModel(nn.Module):
    """EfficientDet full model with pre and post processing."""

    def __init__(
        self,
        compound_coef,
        num_classes,
        ratios,
        scales,
        onnx_export,
    ) -> None:
        super().__init__()

        self.efficientdet_net = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=num_classes,
            ratios=ratios,
            scales=scales,
            onnx_export=onnx_export,
        )
        # self.regress_boxes = BBoxTransform()
        # self.clip_boxes = ClipBoxes()

    def normalize(
        self, inputs, mean_rgb=[0.485, 0.456, 0.406], stddev_rgb=[0.229, 0.224, 0.225]
    ):
        mean = torch.tensor(mean_rgb).cuda()[..., None, None]
        std = torch.tensor(stddev_rgb).cuda()[..., None, None]
        return (inputs / 255 - mean) / std

    def forward(self, inputs):
        inputs = self.normalize(inputs)

        _, boxes, scores, anchors = self.efficientdet_net(inputs)
        anchors = decode_anchors_to_centersize(anchors)

        return boxes, scores, anchors


if __name__ == "__main__":

    cfg = load_yaml(CONFIG_PATH)
    anchors_ratios = eval(cfg["anchors_ratios"])
    anchors_scales = eval(cfg["anchors_scales"])
    obj_list = cfg["obj_list"]

    model = EfficientDetModel(
        compound_coef=COMPOUND_COEF,
        num_classes=len(obj_list),
        # replace this part with your project's anchor config
        ratios=anchors_ratios,
        scales=anchors_scales,
        onnx_export=True,
    )
    model.efficientdet_net.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
    model.eval()
    model.cuda()

    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device="cuda").to(
        torch.float32
    )

    torch.onnx.export(
        model,
        dummy_input,
        "onnx_inference/model/v1/effdet-d2_fp32.onnx",
        input_names=["images"],
        output_names=["boxes", "scores", "anchors"],
        opset_version=13,
    )
