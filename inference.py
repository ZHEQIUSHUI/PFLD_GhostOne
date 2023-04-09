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

dummy_input = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(cfg.DEVICE)
# writer.add_graph(model, (dummy_input,))
# ops, params = thop.profile(model, [dummy_input])
# print(ops/1000000, " mflops", params/1000000, " mparams")

model.load_state_dict(torch.load(cfg.RESUME_MODEL_PATH))

model.eval()

image_paths = glob.glob("/home/qiushui/Pictures/Screenshots/*")

first = False

for ipath in image_paths:
    img_src = cv2.imread(ipath)
    img = transform(img_src).to(device)
    img = img.reshape(-1, *img.shape)
    # print(img.shape)

    if not first:
        # torch.onnx.export(model,[img],"1.onnx")
        ops,params = thop.profile(model,[img])
        print(ops/1000000," mflops",params/1000000," mparams")
        first = True

    landmark_pred, depth_pred, angle_pred = model(img)

    height, width = img_src.shape[:2]
    landmark_pred *= 0.5
    landmark_pred += 0.5
    landmark_pred = landmark_pred.reshape(-1,2)
    landmark_pred[:, 0] *= width
    landmark_pred[:, 1] *= height

    landmark_pred = landmark_pred.cpu().detach().numpy()
    angle_pred = angle_pred.cpu().detach().numpy()
    depth_pred = depth_pred.cpu().detach().numpy()

    landmark_pred = landmark_pred.astype(np.int32)
    for pt in landmark_pred:
        cv2.circle(img_src, pt, 2, (0, 0, 255), 2)

    cv2.imshow("", img_src)
    key = cv2.waitKey(0)
    if key == 27:
        break
