from os import PRIO_PGRP
from tkinter import W
import cv2
import glob
import numpy as np
import random
import cv2
import numpy as np
from torch import Tensor
from torch.utils.data.dataloader import Dataset
from torchvision import transforms
import glob
import albumentations as A
from torchvision.transforms.transforms import ToPILImage


class LMK3DDataSet(Dataset):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.image_lst = glob.glob(
            '/home/qiushui/Datasets/landmark_dataset_new/*/images/*')
        self.input_size = input_size
        self.transform_kps = A.Compose(
            [
                A.OneOf([
                    A.Resize(int(input_size*1.0), int(input_size*1.0)),

                    A.Resize(int(input_size*1.05), int(input_size*1.05)),
                    A.Resize(int(input_size*1.0), int(input_size*1.05)),
                    A.Resize(int(input_size*1.05), int(input_size*1.0)),

                    A.Resize(int(input_size*1.1), int(input_size*1.1)),
                    A.Resize(int(input_size*1.0), int(input_size*1.1)),
                    A.Resize(int(input_size*1.1), int(input_size*1.0)),

                    A.Resize(int(input_size*1.15), int(input_size*1.15)),
                    A.Resize(int(input_size*1.0), int(input_size*1.15)),
                    A.Resize(int(input_size*1.15), int(input_size*1.0)),

                    A.Resize(int(input_size*1.2), int(input_size*1.2)),
                    A.Resize(int(input_size*1.0), int(input_size*1.2)),
                    A.Resize(int(input_size*1.2), int(input_size*1.0)),

                    A.Resize(int(input_size*1.25), int(input_size*1.25)),
                    A.Resize(int(input_size*1.0), int(input_size*1.25)),
                    A.Resize(int(input_size*1.25), int(input_size*1.0)),

                    A.Resize(int(input_size*1.3), int(input_size*1.3)),
                    A.Resize(int(input_size*1.0), int(input_size*1.3)),
                    A.Resize(int(input_size*1.3), int(input_size*1.0)),
                ], p=1),
                A.RGBShift(),
                A.ChannelShuffle(),
                A.RandomBrightness(),
                A.Cutout(max_w_size=int(input_size/8),
                         max_h_size=int(input_size/8)),
                A.GaussianBlur(),
                A.RandomCrop(input_size, input_size, p=1)
            ],
            keypoint_params=A.KeypointParams("xy", remove_invisible=False)
        )

        self.transform = transforms.Compose([
            ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomGrayscale(0.3),
            transforms.RandomAdjustSharpness(3, 0.3),
            transforms.RandomAutocontrast(0.3),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_path = self.image_lst[index]
        label_path = img_path.replace(
            '/images/', '/labels_new/').replace('.jpg', '.npy')
        label = np.load(label_path, allow_pickle=True)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        lmk = label.item().get('lmk')
        angles = label.item().get('angle')

        lmk_xy = lmk[:2].transpose([1, 0])
        lmk_depth = lmk[2]

        lmk_depth /= w
        lmk_depth = (lmk_depth-0.5)/0.5

        val = random.randrange(0, 100)
        if val > 50:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = self.transform_kps(image=img, keypoints=lmk_xy)
        img = res["image"]
        lmk_xy = res["keypoints"]
        lmk_xy = np.array(lmk_xy)

        lmk_xy /= self.input_size
        lmk_xy = (lmk_xy-0.5)/0.5

        angles = angles / 180*3.14

        img = self.transform(img)

        lmk_xy = Tensor(lmk_xy)
        lmk_depth = Tensor(lmk_depth)
        angles = Tensor(angles)
        return img, lmk_xy, lmk_depth, angles

    def __len__(self):
        return len(self.image_lst)


if __name__ == "__main__":
    dataset = LMK3DDataSet(160)

    print(dataset.__len__())
    while 1:
        idx = random.randint(0,len(dataset))
        img, lmk, depth, angles = dataset.__getitem__(idx)

        img = (img*0.5+0.5)*255
        img = img.cpu().detach().numpy().transpose(
            [1, 2, 0]).astype(np.uint8).copy()
        print(img.shape, lmk.shape, depth.shape, angles.shape)

        height, width = img.shape[:2]
        lmk *= 0.5
        lmk += 0.5
        lmk = lmk.reshape(-1, 2)
        lmk[:, 0] *= width
        lmk[:, 1] *= height
        lmk = lmk.cpu().detach().numpy().astype(np.int32)
        for pt in lmk:
            # print(pt)
            cv2.circle(img, pt, 1, (0, 0, 255), 1)

        cv2.imshow("", img)
        key = cv2.waitKey(0)
        if key == 27:
            break
