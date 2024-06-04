#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import argparse
import torchvision.transforms as T
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from models.yolov1 import YOLO_v1
from utils.metrics import evaluate
from utils.engine import postprocess
def get_args_parser():
    parser = argparse.ArgumentParser('Eval Model', add_help=False)
    parser.add_argument('--data_path', default='./datasets/', type=str,help='dataset path')
    parser.add_argument('--input_size', default=[448,448],nargs='+',type=int,help='images input size')
    parser.add_argument('--weights', default='./output_dir/last.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=20, type=int,help='number of the classification types')
    parser.add_argument('--conf_thresh', default=0.01, type=float,help='thresh of cls conf')
    parser.add_argument('--prob_thresh', default=0.01, type=float,help='thresh of predict prob')
    parser.add_argument('--nms_thresh', default=0.5, type=float,help='nms thresh of predict prob')

    parser.add_argument('--grid_size', default=7, type=int,help='grid size of each image')
    parser.add_argument('--num_bboxes', default=2, type=int,help='boxes number of each grid')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')

    return parser


def main(args):

    device = torch.device(args.device)

    VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

    transforms = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    targets = defaultdict(list)
    preds = defaultdict(list)

    model = YOLO_v1(num_classes = args.nb_classes,num_bboxes=args.num_bboxes)

    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    print('Preparing ground-truth data...')

    # Load annotations from label file.
    annotations = []
    with open(os.path.join(args.data_path,'val.txt'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        anno = line.strip().split()
        annotations.append(anno)

    # Prepare ground-truth data.
    image_fnames = []
    for anno in annotations:
        filename = anno[0]
        image_fnames.append(filename)

        num_boxes = (len(anno) - 1) // 5
        for b in range(num_boxes):
            x1 = int(anno[5*b + 1])
            y1 = int(anno[5*b + 2])
            x2 = int(anno[5*b + 3])
            y2 = int(anno[5*b + 4])

            class_label = int(anno[5*b + 5])
            class_name = VOC_CLASSES[class_label]

            targets[(filename, class_name)].append([x1, y1, x2, y2])


    print('Predicting...')

    # Detect objects with the model.
    for filename in tqdm(image_fnames):
        image_path = os.path.join(args.data_path,'JPEGImages',filename)
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        input_tensor = transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)

        boxes, labels, probs = postprocess(output, width, height, VOC_CLASSES, args.grid_size, args.num_bboxes,
                                           args.conf_thresh, args.prob_thresh, args.nms_thresh, args.nb_classes)

        for box, class_name, prob in zip(boxes, labels, probs):
            x1y1, x2y2 = box
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            preds[class_name].append([filename, prob, x1, y1, x2, y2])

    print('Evaluate the detection result...')

    evaluate(preds, targets, class_names=VOC_CLASSES)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
