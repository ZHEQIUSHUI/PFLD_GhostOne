import torch.utils.data as data
import glob, cv2
import numpy as np
import random
import torch
from torchvision import transforms
import albumentations as A

class CocoPose(data.Dataset):
    def __init__(self, root, input_size) -> None:
        super().__init__()
        self.image_list = glob.glob(root + "/images/*.jpg")
        self.input_size = input_size
        self.transform_kps = A.Compose(
            [
                # A.OneOf([
                #     A.Resize(int(input_size*1.0), int(input_size*1.0)),

                #     A.Resize(int(input_size*1.05), int(input_size*1.05)),
                #     A.Resize(int(input_size*1.0), int(input_size*1.05)),
                #     A.Resize(int(input_size*1.05), int(input_size*1.0)),

                #     A.Resize(int(input_size*1.1), int(input_size*1.1)),
                #     A.Resize(int(input_size*1.0), int(input_size*1.1)),
                #     A.Resize(int(input_size*1.1), int(input_size*1.0)),

                #     A.Resize(int(input_size*1.15), int(input_size*1.15)),
                #     A.Resize(int(input_size*1.0), int(input_size*1.15)),
                #     A.Resize(int(input_size*1.15), int(input_size*1.0)),

                #     A.Resize(int(input_size*1.2), int(input_size*1.2)),
                #     A.Resize(int(input_size*1.0), int(input_size*1.2)),
                #     A.Resize(int(input_size*1.2), int(input_size*1.0)),

                #     A.Resize(int(input_size*1.25), int(input_size*1.25)),
                #     A.Resize(int(input_size*1.0), int(input_size*1.25)),
                #     A.Resize(int(input_size*1.25), int(input_size*1.0)),

                #     A.Resize(int(input_size*1.3), int(input_size*1.3)),
                #     A.Resize(int(input_size*1.0), int(input_size*1.3)),
                #     A.Resize(int(input_size*1.3), int(input_size*1.0)),
                # ], p=1),
                A.RGBShift(),
                A.ChannelShuffle(),
                A.RandomBrightness(),
                A.Cutout(max_w_size=int(input_size[0]/8),
                         max_h_size=int(input_size[1]/8)),
                A.GaussianBlur(),
                # A.RandomCrop(input_size, input_size, p=1)
            ]
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomGrayscale(0.3),
            transforms.RandomAdjustSharpness(3, 0.3),
            transforms.RandomAutocontrast(0.3),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        image_path: str = self.image_list[index]
        label_path = image_path.replace("/images/",
                                        "/labels/").replace(".jpg", ".txt")
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        label = np.loadtxt(label_path).reshape(-1, 3)
        kps = label[:, :2]
        visable = label[:, 2]
        visable[visable > 0] = 1

        kps[:, 0] /= width
        kps[:, 1] /= height
        kps -= 0.5
        kps /= 0.5

        img = cv2.resize(img, (self.input_size))
        img = self.transform_kps(image=img)["image"]

        return img, kps, visable

    def __len__(self) -> int:
        return len(self.image_list)


if __name__ == "__main__":
    dataset = CocoPose("/ihoment/junda/pycode/cocokp", (192, 256))
    print(len(dataset))

    for i in range(10):
        img, kps, visable = dataset.__getitem__(random.randint(
            0, len(dataset)))
        print(visable)
        # kps = np.array(label, np.int32).reshape(-1, 3)
        kps *= 0.5
        kps += 0.5
        for kp in kps:
            kp *= (192, 256)
            cv2.circle(img, kp.astype(np.int32), 2, (0, 0, 255), 2)
        cv2.imwrite("%d.jpg" % i, img)
