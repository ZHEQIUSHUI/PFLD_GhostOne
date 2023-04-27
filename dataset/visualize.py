from pycocotools.coco import COCO
import cv2, tqdm
import numpy as np


def show_skelenton(img, kpts, color=(255, 128, 128), thr=0.01):
    kpts = np.array(kpts).reshape(-1, 3)

    for pt in kpts:
        if pt[2] > 0:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, color, 2, 8)
    # skelenton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
    #                 [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # for sk in skelenton:

    #     pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
    #     pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
    #     if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0 and kpts[sk[0] - 1, 2] > thr and kpts[sk[1] - 1, 0] > thr:
    #         cv2.line(img, pos1, pos2, color, 2, 8)
    return img


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

color_list = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE]

root = "/ihoment/public/coco/"
train_kps_name = "person_keypoints_train2017.json"
val_kps_name = "person_keypoints_val2017.json"

coco = COCO(root + "annotations/" + train_kps_name)
id = 0
img_ids = coco.getImgIds(catIds=[1])
for img_idx in img_ids:
    img_name = str(img_idx).zfill(12) + '.jpg'
    img_path = root + 'images/train2017/' + img_name
    img = cv2.imread(img_path)
    annIds = coco.getAnnIds(imgIds=img_idx, iscrowd=False)
    objs = coco.loadAnns(annIds)
    for person_id, obj in enumerate(objs):
        keypoints = np.array(obj['keypoints']).reshape(-1, 3)
        bbox = np.array(obj["bbox"])
        crop = img[int(bbox[1]):int(bbox[1] + bbox[3]),
                   int(bbox[0]):int(bbox[0] + bbox[2]), :].copy()

        x, y, w, h = bbox

        for idx, kp in enumerate(keypoints):
            if kp[2] > 0:
                keypoints[idx][0] -= x
                keypoints[idx][1] -= y
                # cv2.circle(crop,
                #            (int(keypoints[idx][0]), int(keypoints[idx][1])), 2,
                #            (0, 0, 255), 2)

        if np.sum(keypoints) > 0:
            id += 1
            ctx = " ".join(
                [str(item) for item in keypoints.reshape(-1).tolist()])
            cv2.imwrite("images/%08d.jpg" % id, crop)
            f = open("labels/%08d.txt" % id, 'w')
            f.write(ctx)
            f.close()
        # else:
        #     print(keypoints)

        # print(len(keypoints))
        # color = color_list[person_id % len(color_list)]
        # img = show_skelenton(img, keypoints, color=color)
    # save_path = img_name
    # cv2.imwrite(save_path, img)
    # cv2.waitKey(0)
