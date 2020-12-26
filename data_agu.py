import cv2
import numpy as np
import os
from PIL import Image,ImageEnhance
from torch.utils.data import Dataset
import pandas as pd
import random


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    res=(dst-mi)/(0.000000001+(ma-mi))
    res[np.isnan(res)]=0
    return res


class Mydataset(Dataset):
    def __init__(self, path,augment=False,transform=None, target_transform=None):
       
        self.aug=augment
        self.file_path=os.path.dirname(path)
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.iloc[i,0], data.iloc[i,1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        if self.aug==False:
            fn, lab = self.imgs[item]
            # fn = os.path.join(self.file_path, "image_A/" + fn)
            # label = os.path.join(self.file_path, "image_A/" + lab)
            fn = os.path.join(self.file_path, "images/"+ fn)
            label = os.path.join(self.file_path, "labels/"+ lab)

            bgr_img = cv2.imread(fn, -1)
            rgb_img = bgr_img[..., ::-1]  # bgr2rgb
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            grad = (255 * grade(gray)).astype(np.uint8)

            # img = Image.open(fn).convert('RGB')
            img = cv2.merge([rgb_img, grad])
            img = Image.fromarray(img, mode="CMYK")
            if self.transform is not None:
                img = self.transform(img)

            gt = cv2.imread(label, -1)
            return img, gt, lab


        else:
            # 进行数据增强
            fn, lab = self.imgs[item]
            # train with data.cvs
            fn = os.path.join(self.file_path, "images/"+ fn)
            label = os.path.join(self.file_path, "labels/"+ lab)

            gt = cv2.imread(label, -1)
            image = cv2.imread(fn,-1)

            image = randomHueSaturationValue(image,
                                             hue_shift_limit=(-30, 30),
                                             sat_shift_limit=(-5, 5),
                                             val_shift_limit=(-15, 15))

            image, gt = randomShiftScaleRotate(image, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))

            image, gt = randomHorizontalFlip(image, gt)
            image, gt = randomVerticleFlip(image, gt)
            image, gt = randomRotate90(image, gt)

            rgb_img = image[..., ::-1]  # bgr2rgb
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            grad = (255 * grade(gray)).astype(np.uint8)
            img = cv2.merge([rgb_img, grad])
            img = Image.fromarray(img, mode="CMYK")
            if self.transform is not None:
                img = self.transform(img.copy())
            return img, gt.copy(), lab

    def __len__(self):
        return len(self.imgs)

