from torchvision import transforms as trans
from easydict import EasyDict as edict
from utils.utils import get_time
import os
import torch


def get_config(isTrain = True) -> edict:
    cfg = edict()
    cfg.SEED = 2023
    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.GPU_ID = 0

    cfg.TRANSFORM = trans.Compose([trans.ToTensor(),
                                   trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    cfg.MODEL_TYPE = 'PFLD_GhostNet_Slim_3D'  # [PFLD, PFLD_GhostNet, PFLD_GhostNet_Slim, PFLD_GhostOne]
    cfg.INPUT_SIZE = [128, 128]
    cfg.WIDTH_FACTOR = 1
    cfg.LANDMARK_NUMBER = 68

    cfg.TRAIN_BATCH_SIZE = 64
    cfg.VAL_BATCH_SIZE = 8

    cfg.TRAIN_DATA_PATH = './data/train_data_repeat80/list.txt'
    cfg.VAL_DATA_PATH = './data/test_data_repeat80/list.txt'

    cfg.EPOCHES = 80
    cfg.LR = 1e-7
    cfg.WEIGHT_DECAY = 1e-6
    cfg.NUM_WORKERS = 11
    cfg.MILESTONES = [20, 45]

    cfg.RESUME = True
    if cfg.RESUME:
        cfg.RESUME_MODEL_PATH = "checkpoint/models/PFLD_GhostNet_Slim_3D_1_128_2023-04-09-11-47/pfld_ghostnet_slim_3d_step:12.pth"

    create_time = get_time()
    cfg.MODEL_PATH = './checkpoint/models/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOG_PATH = './checkpoint/log/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOGGER_PATH = os.path.join(cfg.MODEL_PATH, "train.log")
    if isTrain and not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    return cfg
