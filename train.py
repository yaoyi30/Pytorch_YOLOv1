#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import os
import torch
import torch.nn as nn
from models.yolov1 import YOLO_v1
import argparse
import numpy as np
from utils.transform import Resize,Compose,ToTensor,Normalize,RandomHorizontalFlip,RandomVerticalFlip,RandomScale,\
                            RandomHue,RandomSaturation,RandomBrightness,RandomGaussianBlur,RandomCrop,RandomShift
from utils.datasets import DetData
from utils.loss import Detect_Loss
from utils.engine import train_and_val,plot_loss,plot_lr

def get_args_parser():
    parser = argparse.ArgumentParser('Image Detection Train', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,help='Batch size for training')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--input_size', default=[448,448],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default='./datasets/', type=str,help='dataset path')

    parser.add_argument('--init_lr', default=0.001, type=float,help='SGD intial lr')
    parser.add_argument('--momentum', default=0.9, type=float,help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float,help='SGD weight decay')

    parser.add_argument('--finetune', default='./weights/resnet50_ram-a26f946b.pth',
                        help='finetune from checkpoint')

    parser.add_argument('--nb_classes', default=20, type=int,help='number of the classification types')
    parser.add_argument('--grid_size', default=7, type=int,help='grid size of each image')
    parser.add_argument('--num_bboxes', default=2, type=int,help='boxes number of each grid')

    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser



def main(args):

    device = torch.device(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_transform = Compose([
                                    RandomHorizontalFlip(0.5),
                                    RandomVerticalFlip(0.5),
                                    RandomScale(0.5),
                                    RandomGaussianBlur(0.5),
                                    RandomBrightness(0.5),
                                    RandomHue(0.5),
                                    RandomSaturation(0.5),
                                    RandomShift(0.5),
                                    RandomCrop(0.5),
                                    Resize(args.input_size),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])

    val_transform = Compose([
                                    Resize(args.input_size),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_dataset = DetData(    image_path = os.path.join(args.data_path, 'JPEGImages'),
                                label_file = os.path.join(args.data_path, 'train.txt'),
                                nb_classes = args.nb_classes,
                                grid_size = args.grid_size,
                                num_bboxes = args.num_bboxes,
                                transform = train_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    val_dataset = DetData(    image_path = os.path.join(args.data_path, 'JPEGImages'),
                              label_file = os.path.join(args.data_path, 'val.txt'),
                              nb_classes = args.nb_classes,
                              grid_size=args.grid_size,
                              num_bboxes=args.num_bboxes,
                              transform = val_transform)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    model = YOLO_v1(num_classes = args.nb_classes,num_bboxes=args.num_bboxes)
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)

    # 多GPU训练使用
    # model = nn.DataParallel(model,[0,1,2,3])

    loss_function = Detect_Loss(feature_size=args.grid_size, num_bboxes=args.num_bboxes, num_classes=args.nb_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(args.epochs * 0.3), gamma=0.1)

    history = train_and_val(args.epochs, model, train_loader,val_loader,loss_function, optimizer,scheduler,args.output_dir,device)

    plot_loss(np.arange(0,args.epochs),args.output_dir, history)
    plot_lr(np.arange(0,args.epochs),args.output_dir, history)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
