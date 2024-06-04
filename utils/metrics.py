#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import numpy as np

def compute_average_precision(recall, precision):

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i -1], precision[i])

    ap = 0.0
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap


def evaluate(preds,targets,class_names,threshold=0.5):

    aps = []

    for class_name in class_names:
        class_preds = preds[class_name]

        if len(class_preds) == 0:
            ap = 0.0
            print('---class {} AP {}---'.format(class_name, ap))
            aps.append(ap)
            break

        image_fnames = [pred[0] for pred in class_preds]
        probs = [pred[1]  for pred in class_preds]
        boxes = [pred[2:] for pred in class_preds]

        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes = [boxes[i] for i in sorted_idxs]

        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == class_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        num_detections = len(boxes)
        tp = np.zeros(num_detections)
        fp = np.ones(num_detections)

        for det_idx, (filename, box) in enumerate(zip(image_fnames, boxes)):

            if (filename, class_name) in targets:
                boxes_gt = targets[(filename, class_name)]
                for box_gt in boxes_gt:

                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if iou >= threshold:
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt)
                        if len(boxes_gt) == 0:
                            del targets[(filename, class_name)]
                        break

            else:
                pass

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = compute_average_precision(recall, precision)
        print('---class {} AP {}---'.format(class_name, ap))
        aps.append(ap)

    print('---mAP {}---'.format(np.mean(aps)))

    return aps