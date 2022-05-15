import torch
import torch.onnx

from backbone import EfficientDetBackbone
from utils.eff_utils import load_yaml

# from utils.utils import preprocess

COMPOUND_COEF = 2
BATCH_SIZE = 1
INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
IMG_SIZE = INPUT_SIZES[COMPOUND_COEF]

CONFIG_PATH = "onnx_inference/projects/0509split.yml"
CKPT_PATH = f"logs/weights/efficientdet-d{COMPOUND_COEF}_125_253500.pth"

cfg = load_yaml(CONFIG_PATH)
anchors_ratios = eval(cfg["anchors_ratios"])
anchors_scales = eval(cfg["anchors_scales"])
obj_list = cfg["obj_list"]

model = EfficientDetBackbone(
    compound_coef=COMPOUND_COEF,
    num_classes=len(obj_list),
    # replace this part with your project's anchor config
    ratios=anchors_ratios,
    scales=anchors_scales,
    onnx_export=True,
)
model.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
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
    opset_version=11,
)
