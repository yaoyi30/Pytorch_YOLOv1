#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

from torchvision.transforms import functional as F
import random
import torch
import numpy as np
import cv2
from PIL import Image

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image,boxes,labels):
        width, height = image.size
        image = F.resize(image, self.size)

        scale_x = self.size[1] / width
        scale_y = self.size[0] / height

        scale_tensor = torch.FloatTensor([[scale_x, scale_y, scale_x, scale_y]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        
        return image,boxes,labels


class RandomGaussianBlur(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, boxes,labels):
        if random.random() < self.prob:
            ksize = random.choice([3, 5])
            image = F.gaussian_blur(image,[ksize, ksize])

        return image, boxes,labels


class RandomBrightness(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            adjust = random.uniform(0.5, 1.5)
            image = F.adjust_brightness(image,adjust)

        return image,boxes,labels


class RandomHue(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            adjust = random.uniform(-0.5, 0.5)
            image = F.adjust_hue(image,adjust)

        return image,boxes,labels


class RandomSaturation(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            adjust = random.uniform(0.5, 1.5)
            image = F.adjust_saturation(image,adjust)

        return image,boxes,labels


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            width, height = image.size
            image = F.hflip(image)
            x1, x2 = boxes[:, 0], boxes[:, 2]
            x1_new = width - x2
            x2_new = width - x1
            boxes[:, 0], boxes[:, 2] = x1_new, x2_new

        return image,boxes,labels

class RandomScale(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            width, height = image.size
            scale = random.uniform(0.8,1.2)
            image = F.resize(image,[height,int(width*scale)])

            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor

        return image,boxes,labels

class RandomShift(object):
    def __init__(self, prob):
        self.prob = prob
        self.mean = [122.67891434, 116.66876762, 104.00698793]

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2.0
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            h, w, c = img.shape
            img_out = np.zeros((h, w, c), dtype=img.dtype)
            mean_bgr = self.mean[::-1]
            img_out[:, :] = mean_bgr

            dx = random.uniform(-w * 0.2, w * 0.2)
            dy = random.uniform(-h * 0.2, h * 0.2)
            dx, dy = int(dx), int(dy)

            if dx >= 0 and dy >= 0:
                img_out[dy:, dx:] = img[:h - dy, :w - dx]
            elif dx >= 0 and dy < 0:
                img_out[:h + dy, dx:] = img[-dy:, :w - dx]
            elif dx < 0 and dy >= 0:
                img_out[dy:, :w + dx] = img[:h - dy, -dx:]
            elif dx < 0 and dy < 0:
                img_out[:h + dy, :w + dx] = img[-dy:, -dx:]

            center = center + torch.FloatTensor([[dx, dy]]).expand_as(center)  # [n, 2]
            mask_x = (center[:, 0] >= 0) & (center[:, 0] < w)  # [n,]
            mask_y = (center[:, 1] >= 0) & (center[:, 1] < h)  # [n,]
            mask = (mask_x & mask_y).view(-1, 1)  # [n, 1], mask for the boxes within the image after shift.

            boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4)  # [m, 4]
            if len(boxes_out) == 0:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), boxes, labels
            shift = torch.FloatTensor([[dx, dy, dx, dy]]).expand_as(boxes_out)  # [m, 4]

            boxes_out = boxes_out + shift
            boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
            boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
            boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
            boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

            labels_out = labels[mask.view(-1)]

            image, boxes, labels = Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)), boxes_out, labels_out

        return image,boxes,labels

class RandomCrop(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

            w_orig, h_orig = image.size
            h = random.uniform(0.6 * h_orig, h_orig)
            w = random.uniform(0.6 * w_orig, w_orig)
            y = random.uniform(0, h_orig - h)
            x = random.uniform(0, w_orig - w)
            h, w, x, y = int(h), int(w), int(x), int(y)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)  # [n, 2]
            mask_x = (center[:, 0] >= 0) & (center[:, 0] < w)  # [n,]
            mask_y = (center[:, 1] >= 0) & (center[:, 1] < h)  # [n,]
            mask = (mask_x & mask_y).view(-1, 1)  # [n, 1], mask for the boxes within the image after crop.

            boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4)  # [m, 4]
            if len(boxes_out) == 0:
                return image, boxes, labels
            shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_out)  # [m, 4]

            boxes_out = boxes_out - shift
            boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
            boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
            boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
            boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

            labels_out = labels[mask.view(-1)]
            box = (x, y, x + w, y + h)
            img_out = image.crop(box)

            image, boxes, labels = img_out,boxes_out,labels_out

        return image,boxes,labels

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image,boxes,labels):
        if random.random() < self.prob:
            width, height = image.size
            image = F.vflip(image)
            y1, y2 = boxes[:, 1], boxes[:, 3]
            y1_new = height - y2
            y2_new = height - y1
            boxes[:, 1], boxes[:, 3] = y1_new, y2_new

        return image,boxes,labels

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image,boxes,labels):
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image,boxes,labels

class ToTensor(object):
    def __call__(self, image,boxes,labels):
        image = F.to_tensor(image)

        return image,boxes,labels


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image,boxes,labels):
        for t in self.transforms:
            image,boxes,labels = t(image,boxes,labels)

        return image,boxes,labels
