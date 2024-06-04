#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import os
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_and_val(epochs, model, train_loader, val_loader,criterion, optimizer,scheduler,output_dir,device):

    train_loss = []
    val_loss = []
    learning_rate = []
    best_min_loss = 100

    model.to(device)

    fit_time = time.time()
    for e in range(epochs):

        torch.cuda.empty_cache()
        print("This Epoch Learning Rate: {:.6f}  ".format(scheduler.get_last_lr()[0]))
        since = time.time()
        training_loss = 0

        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:

                image = image.to(device)
                label = label.to(device)

                output = model(image)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                pbar.update(1)

        model.eval()
        validation_loss = 0

        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:

                    image = image.to(device)
                    label = label.to(device)

                    output = model(image)
                    loss = criterion(output, label)

                    validation_loss += loss.item()
                    pb.update(1)

        train_loss.append(training_loss / len(train_loader))
        val_loss.append(validation_loss / len(val_loader))

        learning_rate.append(scheduler.get_last_lr())
        
        save_file = open(os.path.join(output_dir,'log.txt'), mode='a+')
        save_file.writelines(["Epoch:{}/{}  ".format(e + 1, epochs)+
              "Learning Rate: {:.6f}  ".format(scheduler.get_last_lr()[0]) +
              "Train Loss: {:.3f}  ".format(training_loss / len(train_loader))+
              "Val Loss: {:.3f}  ".format(validation_loss / len(val_loader))+'\n'])
        save_file.close()

        # 多GPU训练使用
        # torch.save(model.module.state_dict(), os.path.join(output_dir,'last.pth'))
        torch.save(model.state_dict(), os.path.join(output_dir,'last.pth'))
        if best_min_loss > (validation_loss / len(val_loader)):
            print("--save best model,loss is {:.6f}--".format(validation_loss / len(val_loader)))
            best_min_loss = validation_loss / len(val_loader)
            # 多GPU训练使用
            # torch.save(model.module.state_dict(), os.path.join(output_dir,'best.pth'))
            torch.save(model.state_dict(), os.path.join(output_dir,'best.pth'))

        print("Epoch:{}/{} ".format(e + 1, epochs),
              "Train Loss: {:.3f} ".format(training_loss / len(train_loader)),
              "Val Loss: {:.3f} ".format(validation_loss / len(val_loader)),
              "Time: {:.2f}s".format((time.time() - since)))

        scheduler.step()

    history = {'train_loss': train_loss, 'val_loss': val_loss ,'lr':learning_rate}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history


def plot_loss(x,output_dir, history):
    plt.plot(x, history['val_loss'], label='val', marker='o')
    plt.plot(x, history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(output_dir,'loss.png'))
    plt.clf()

def plot_lr(x,output_dir,  history):
    plt.plot(x, history['lr'], label='learning_rate', marker='x')
    plt.title('learning rate per epoch')
    plt.ylabel('Learning_rate')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(output_dir,'learning_rate.png'))
    plt.clf()


def nms(boxes, scores,threshold):

    x1 = boxes[:, 0]  # [n,]
    y1 = boxes[:, 1]  # [n,]
    x2 = boxes[:, 2]  # [n,]
    y2 = boxes[:, 3]  # [n,]
    areas = (x2 - x1) * (y2 - y1)  # [n,]

    _, ids_sorted = scores.sort(0, descending=True)  # [n,]
    ids = []
    while ids_sorted.numel() > 0:
        # Assume `ids_sorted` size is [m,] in the beginning of this iter.

        i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        ids.append(i)

        if ids_sorted.numel() == 1:
            break  # If only one box is left (i.e., no box to supress), break.

        inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i])  # [m-1, ]
        inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i])  # [m-1, ]
        inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i])  # [m-1, ]
        inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
        inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [m-1, ]

        inters = inter_w * inter_h  # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        unions = areas[i] + areas[ids_sorted[1:]] - inters  # unions b/w/ box `i` and other boxes, sized [m-1, ].
        ious = inters / unions  # [m-1, ]

        # Remove boxes whose IoU is higher than the threshold.
        ids_keep = (ious <= threshold).nonzero().squeeze()  # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        if ids_keep.numel() == 0:
            break  # If no box left, break.
        ids_sorted = ids_sorted[ids_keep + 1]  # `+1` is needed because `ids_sorted[0] = i`.

    return torch.LongTensor(ids)


def decode(pred_tensor,grid_size,num_bboxes,conf_thresh,prob_thresh,nb_classes):

    S, B, C = grid_size,num_bboxes,nb_classes
    boxes, labels, confidences, class_scores = [], [], [], []

    cell_size = 1.0 / float(S)

    pred_tensor = pred_tensor.cpu().data.squeeze(0)

    pred_tensor_conf_list = []
    for b in range(B):
        pred_tensor_conf_list.append(pred_tensor[:, :, 5 * b + 4].unsqueeze(2))
    grid_ceil_conf = torch.cat(pred_tensor_conf_list, 2)

    grid_ceil_conf, grid_ceil_index = grid_ceil_conf.max(2)
    class_conf, class_index = pred_tensor[:, :, 5 * B:].max(2)
    class_conf[class_conf <= conf_thresh] = 0
    class_prob = class_conf * grid_ceil_conf

    for i in range(S):
        for j in range(S):
            if float(class_prob[j, i]) < prob_thresh:
                continue
            box = pred_tensor[j, i, 5 * grid_ceil_index[j, i]: 5 * grid_ceil_index[j, i] + 4]
            xy_start_pos = torch.FloatTensor([i, j]) * cell_size
            xy_normalized = box[:2] * cell_size + xy_start_pos
            wh_normalized = box[2:]
            box_xyxy = torch.FloatTensor(4)
            box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized
            box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized

            boxes.append(box_xyxy)
            labels.append(class_index[j, i])
            confidences.append(grid_ceil_conf[j, i])
            class_scores.append(class_conf[j, i])

    if len(boxes) > 0:
        boxes = torch.stack(boxes, 0)
        labels = torch.stack(labels, 0)
        confidences = torch.stack(confidences, 0)
        class_scores = torch.stack(class_scores, 0)
    else:
        boxes = torch.FloatTensor(0, 4)
        labels = torch.LongTensor(0)
        confidences = torch.FloatTensor(0)
        class_scores = torch.FloatTensor(0)

    return boxes, labels, confidences, class_scores

def postprocess(output,width, height,VOC_CLASSES,grid_size,num_bboxes,conf_thresh,prob_thresh,nms_thresh,nb_classes):

    boxes,labels,probs = [],[],[]

    boxes_list, labels_list, confidences_list, class_scores_list = decode(output, grid_size, num_bboxes,
                                                                          conf_thresh, prob_thresh,
                                                                          nb_classes)
    if boxes_list.shape[0] != 0:
        boxes_nms, labels_nms, probs_nms = [], [], []
        for class_label in range(len(VOC_CLASSES)):
            ids = (labels_list == class_label)
            if torch.sum(ids) == 0:
                continue

            boxes_list_current_cls = boxes_list[ids]
            labels_list_current_cls = labels_list[ids]
            confidences_list_current_cls = confidences_list[ids]
            class_scores_list_current_cls = class_scores_list[ids]

            ids_postprocess = nms(boxes_list_current_cls, confidences_list_current_cls, nms_thresh)

            boxes_nms.append(boxes_list_current_cls[ids_postprocess])
            labels_nms.append(labels_list_current_cls[ids_postprocess])
            probs_nms.append(
                confidences_list_current_cls[ids_postprocess] * class_scores_list_current_cls[ids_postprocess])

        boxes_nms = torch.cat(boxes_nms, 0)
        labels_nms = torch.cat(labels_nms, 0)
        probs_nms = torch.cat(probs_nms, 0)

        for box, label, prob in zip(boxes_nms, labels_nms, probs_nms):
            x1, x2 = width * box[0], width * box[2]  # unnormalize x with image width.
            y1, y2 = height * box[1], height * box[3]  # unnormalize y with image height.
            boxes.append(((x1, y1), (x2, y2)))

            label_idx = int(label)  # convert from LongTensor to int.
            class_name = VOC_CLASSES[label_idx]
            labels.append(class_name)

            prob = float(prob)
            probs.append(prob)

    return boxes,labels,probs
