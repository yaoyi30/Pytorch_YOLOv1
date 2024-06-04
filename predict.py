#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import argparse
import torch
import numpy as np
import torchvision.transforms as T
from models.yolov1 import YOLO_v1
from PIL import Image
import cv2
from utils.engine import postprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Predict Image', add_help=False)
    parser.add_argument('--image_path', default='./people.jpg', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument('--input_size', default=[448,448],nargs='+',type=int,help='images input size')
    parser.add_argument('--weights', default='./output_dir/last.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=20, type=int,help='number of the classification types')
    parser.add_argument('--conf_thresh', default=0.1, type=float,help='thresh of cls conf')
    parser.add_argument('--prob_thresh', default=0.1, type=float,help='thresh of predict prob')
    parser.add_argument('--nms_thresh', default=0.5, type=float,help='nms thresh of predict prob')

    parser.add_argument('--grid_size', default=7, type=int,help='grid size of each image')
    parser.add_argument('--num_bboxes', default=2, type=int,help='boxes number of each grid')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')

    return parser

def main(args):
    device = torch.device(args.device)

    image = Image.open(args.image_path).convert('RGB')
    width, height = image.size

    VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

    transforms = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    model = YOLO_v1(num_classes = args.nb_classes,num_bboxes=args.num_bboxes)

    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    input_tensor = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    boxes, labels, probs = postprocess(output,width, height, VOC_CLASSES,args.grid_size, args.num_bboxes,
                                       args.conf_thresh,args.prob_thresh,args.nms_thresh,args.nb_classes)

    cv_image = np.array(image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    for box, label, prob in zip(boxes, labels, probs):
        (left,top),(right,bottom) = box
        cv2.rectangle(cv_image, (int(left), int(top)), (int(right), int(bottom)), (128, 128, 0), thickness=2)
        cv2.putText(cv_image, label+' '+'{:.2f}'.format(prob), (int(left),  int(top)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    cv2.imwrite('result.png',cv_image)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
