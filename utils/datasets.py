#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import torch
import os
import os.path
from torch.utils.data import Dataset
from PIL import Image

class DetData(Dataset):
    def __init__(self, image_path, label_file, nb_classes,grid_size = 7,num_bboxes = 2,transform = None):
        self.image_path = image_path
        self.label_file = label_file
        self.transform = transform

        self.images = []
        self.boxes = []
        self.labels = []

        self.S = grid_size
        self.B = num_bboxes
        self.C = nb_classes

        file_txt = open(self.label_file)
        lines = file_txt.readlines()
        for line in lines:
            information = line.strip().split()
            self.images.append(information[0])
            num_boxes = (len(information) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = float(information[1 + 5 * i])
                ymin = float(information[2 + 5 * i])
                xmax = float(information[3 + 5 * i])
                ymax = float(information[4 + 5 * i])
                cls = int(information[5 + 5 * i])
                box.append([xmin,ymin,xmax,ymax])
                label.append(cls)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __len__(self):
        return len(self.images)

    def encode(self, boxes, labels):

        target = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        cell_size = 1.0 / float(self.S)

        boxes_wh = boxes[:, 2:] - boxes[:, :2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        for xy, wh, label in zip(boxes_xy, boxes_wh, labels):

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])
            x0y0 = ij * cell_size
            xy_offset = (xy - x0y0) / cell_size

            for b in range(self.B):
                s = 5 * b
                target[j, i, s : s+2] = xy_offset
                target[j, i, s+2 : s+4] = wh
                target[j, i, s+4] = 1.0

            target[j, i, label + self.B * 5] = 1.0

        return target

    def __getitem__(self, idx):
        image_filename = self.images[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        image = Image.open(os.path.join(self.image_path, image_filename)).convert('RGB')

        if self.transform is not None:
            image,boxes,labels = self.transform(image,boxes,labels)

        _,height,width = image.shape

        boxes /= torch.Tensor([width, height,width, height]).expand_as(boxes)
        target = self.encode(boxes, labels)

        return image,target