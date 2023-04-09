import torch
from config import get_config
import numpy as np
import thop
from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostNet_Slim_3D import PFLD_GhostNet_Slim_3D
from models.PFLD_GhostNet_Slim_3D_new import PFLD_GhostNet_Slim_3D_new
from models.PFLD_GhostOne import PFLD_GhostOne
from torchvision import transforms
import glob
import cv2
from torchvision.transforms.transforms import ToPILImage


MODEL_DICT = {'PFLD': PFLD,
              'PFLD_GhostNet': PFLD_GhostNet,
              'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
              'PFLD_GhostOne': PFLD_GhostOne,
              'PFLD_GhostNet_Slim_3D': PFLD_GhostNet_Slim_3D,
              'PFLD_GhostNet_Slim_3D_new': PFLD_GhostNet_Slim_3D_new
              }


cfg = get_config(False)
TRAIN_DATA_PATH = cfg.TRAIN_DATA_PATH
VAL_DATA_PATH = cfg.VAL_DATA_PATH
TRANSFORM = cfg.TRANSFORM
MODEL_TYPE = cfg.MODEL_TYPE
WIDTH_FACTOR = cfg.WIDTH_FACTOR
INPUT_SIZE = cfg.INPUT_SIZE
LANDMARK_NUMBER = cfg.LANDMARK_NUMBER

transform = transforms.Compose([
    ToPILImage(),
    transforms.Resize((INPUT_SIZE[0], INPUT_SIZE[0])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

model = MODEL_DICT[MODEL_TYPE](
    WIDTH_FACTOR, INPUT_SIZE[0], LANDMARK_NUMBER).to(device)

dummy_input = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(device)
# writer.add_graph(model, (dummy_input,))

model.load_state_dict(torch.load(cfg.RESUME_MODEL_PATH))
print(cfg.RESUME_MODEL_PATH)

model.eval()
model(dummy_input)

torch.onnx.export(model,dummy_input,"pfld_ghostnet_slim_3d_last.onnx",opset_version=11,input_names=["image"],output_names=["landmark","depth","angle"])
ops, params = thop.profile(model, [dummy_input])
print(ops/1000000, " mflops", params/1000000, " mparams")
