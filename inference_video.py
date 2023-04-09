import torch
from config import get_config
import numpy as np
import thop
from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostNet_Slim_3D import PFLD_GhostNet_Slim_3D
from models.PFLD_GhostOne import PFLD_GhostOne
from torchvision import transforms
import glob
import cv2
from torchvision.transforms.transforms import ToPILImage
import insightface.app as app
import numpy


MODEL_DICT = {'PFLD': PFLD,
              'PFLD_GhostNet': PFLD_GhostNet,
              'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
              'PFLD_GhostOne': PFLD_GhostOne,
              'PFLD_GhostNet_Slim_3D': PFLD_GhostNet_Slim_3D,
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
model.load_state_dict(torch.load(cfg.RESUME_MODEL_PATH))

model.eval()

first = False

cap = cv2.VideoCapture(0)

detector = app.FaceAnalysis("buffalo_sc", allowed_modules=[
                            "detection"], input_size=640)


def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])


def get_face(detector: app.FaceAnalysis, img):
    res = detector.get(img)
    if len(res) > 0:
        bbox = res[0].bbox
        xmin, ymin, xmax, ymax = bbox
        tmp = ((xmax - xmin)/2)*1.2
        xmin_new = (xmin+xmax)/2 - tmp
        xmax_new = (xmin+xmax)/2 + tmp
        ymin_new = (ymin+ymax)/2 - tmp
        ymax_new = (ymin+ymax)/2 + tmp




while 1:
    ret, img_src = cap.read()
    if not ret:
        break

    get_face(detector, img_src)

    img = transform(img_src).to(device)
    img = img.reshape(-1, *img.shape)
    # print(img.shape)

    if not first:
        ops, params = thop.profile(model, [img])
        print(ops/1000000, " mflops", params/1000000, " mparams")
        first = True

    landmark_pred, depth_pred, angle_pred = model(img)

    height, width = img_src.shape[:2]
    landmark_pred *= 0.5
    landmark_pred += 0.5
    landmark_pred = landmark_pred.reshape(-1, 2)
    landmark_pred[:, 0] *= width
    landmark_pred[:, 1] *= height

    landmark_pred = landmark_pred.cpu().detach().numpy()
    angle_pred = angle_pred.cpu().detach().numpy()
    depth_pred = depth_pred.cpu().detach().numpy()

    landmark_pred = landmark_pred.astype(np.int32)
    for pt in landmark_pred:
        cv2.circle(img_src, pt, 2, (0, 0, 255), 2)

    cv2.imshow("", img_src)
    key = cv2.waitKey(1)
    if key == 27:
        break
