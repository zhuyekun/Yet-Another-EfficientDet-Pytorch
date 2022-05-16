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

CONFIG_PATH = "onnx_inference/projects/0509split.yml"
CKPT_PATH = f"logs/weights/efficientdet-d{COMPOUND_COEF}_125_253500.pth"


class FullModel(nn.Module):
    def __init__(
        self,
        compound_coef,
        num_classes,
        ratios,
        scales,
        onnx_export,
    ) -> None:
        super().__init__()

        self.backbone = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=num_classes,
            ratios=ratios,
            scales=scales,
            onnx_export=onnx_export,
        )
        self.regress_boxes = BBoxTransform()
        self.clip_boxes = ClipBoxes()

    def forward(self, x):
        _, regression, classification, anchors = self.backbone(x)
        transformed_anchors = self.regress_boxes(anchors, regression)
        transformed_anchors = self.clip_boxes(transformed_anchors, x)

        return classification, transformed_anchors


cfg = load_yaml(CONFIG_PATH)
anchors_ratios = eval(cfg["anchors_ratios"])
anchors_scales = eval(cfg["anchors_scales"])
obj_list = cfg["obj_list"]

model = FullModel(
    compound_coef=COMPOUND_COEF,
    num_classes=len(obj_list),
    # replace this part with your project's anchor config
    ratios=anchors_ratios,
    scales=anchors_scales,
    onnx_export=True,
)
model.backbone.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
model.eval()
model.cuda()

dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device="cuda").to(
    torch.float32
)

torch.onnx.export(
    model,
    dummy_input,
    "onnx_inference/model/efficientdet-d2.onnx",
    input_names=["images"],
    opset_version=13,
)
