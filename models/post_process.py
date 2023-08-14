import torch

from .bbox_post_process import iou3d_nms_utils

def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config["NMS_PRE_MAXSIZE"], box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config["NMS_TYPE"])(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config["NMS_THRESH"], **nms_config
        )
        selected = indices[keep_idx[:nms_config["NMS_POST_MAXSIZE"]]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config["NMS_PRE_MAXSIZE"], box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config["NMS_TYPE"])(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config["NMS_THRESH"], **nms_config
            )
            selected = indices[keep_idx[:nms_config["NMS_POST_MAXSIZE"]]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes


def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
    if 'gt_boxes' not in data_dict:
        return recall_dict
    # if data_dict["Model_mode"] == 'train':
    #     return recall_dict
    rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
    #gt_boxes = data_dict['gt_boxes'][batch_index][:,2:9]
    gt_boxes = data_dict['gt_boxes'][batch_index]

    if recall_dict.__len__() == 0:
        recall_dict = {'gt': 0}
        for cur_thresh in thresh_list:
            recall_dict['roi_%s' % (str(cur_thresh))] = 0
            recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

    cur_gt = gt_boxes
    k = cur_gt.__len__() - 1
    while k > 0 and cur_gt[k].sum() == 0:
        k -= 1
    cur_gt = cur_gt[:k + 1]

    if cur_gt.shape[0] > 0:
        if box_preds.shape[0] > 0:
            iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
        else:
            iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

        if rois is not None:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

        for cur_thresh in thresh_list:
            if iou3d_rcnn.shape[0] == 0:
                recall_dict['rcnn_%s' % str(cur_thresh)] += 0
            else:
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
            if rois is not None:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

        recall_dict['gt'] += cur_gt.shape[0]
    else:
        gt_iou = box_preds.new_zeros(box_preds.shape[0])
    return recall_dict

def post_processing(batch_dict,cfg_post_process,num_class):
    """
    Args:
        batch_dict:
            batch_size:
            batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                            or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
            multihead_label_mapping: [(num_class1), (num_class2), ...]
            batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
            cls_preds_normalized: indicate whether batch_cls_preds is normalized
            batch_index: optional (N1+N2+...)
            has_class_labels: True/False
            roi_labels: (B, num_rois)  1 .. num_classes
            batch_pred_labels: (B, num_boxes, 1)
    Returns:
    """
    post_process_cfg = cfg_post_process
    batch_size = batch_dict['batch_cls_preds'].size()[0]
    recall_dict = {}
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_box_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

        box_preds = batch_dict['batch_box_preds'][batch_mask]
        #print("box shape:",box_preds.shape)
        src_box_preds = box_preds

        if not isinstance(batch_dict['batch_cls_preds'], list):
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_cls_preds = cls_preds
            assert cls_preds.shape[1] in [1, num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
        else:
            cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
            src_cls_preds = cls_preds
            if not batch_dict['cls_preds_normalized']:
                cls_preds = [torch.sigmoid(x) for x in cls_preds]
        #print("cls shape:",cls_preds.shape)

        if post_process_cfg["NMS_CONFIG"]["MULTI_CLASSES_NMS"]:
            if not isinstance(cls_preds, list):
                cls_preds = [cls_preds]
                multihead_label_mapping = [torch.arange(1, num_class, device=cls_preds[0].device)]
            else:
                multihead_label_mapping = batch_dict['multihead_label_mapping']

            cur_start_idx = 0
            pred_scores, pred_labels, pred_boxes = [], [], []
            for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                cur_pred_scores, cur_pred_labels, cur_pred_boxes = multi_classes_nms(
                    cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                    nms_config=post_process_cfg["NMS_CONFIG"],
                    score_thresh=post_process_cfg["SCORE_THRESH"]
                )
                cur_pred_labels = cur_label_mapping[cur_pred_labels]
                pred_scores.append(cur_pred_scores)
                pred_labels.append(cur_pred_labels)
                pred_boxes.append(cur_pred_boxes)
                cur_start_idx += cur_cls_preds.shape[0]

            final_scores = torch.cat(pred_scores, dim=0)
            final_labels = torch.cat(pred_labels, dim=0)
            final_boxes = torch.cat(pred_boxes, dim=0)
        else:
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)

            if batch_dict.get('has_class_labels', False):
                label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                label_preds = batch_dict[label_key][index]
            else:
                label_preds = label_preds + 1
            selected, selected_scores = class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=post_process_cfg["NMS_CONFIG"],
                score_thresh=post_process_cfg["SCORE_THRESH"]
            )
            #print("selected :",selected.shape)
            #print("selected_scores :",selected_scores.shape)
            

            if post_process_cfg["OUTPUT_RAW_SCORE"]:
                max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                selected_scores = max_cls_preds[selected]

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]


        recall_dict = generate_recall_record(
            box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
            recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
            thresh_list=post_process_cfg["RECALL_THRESH_LIST"]
        )

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)

    return pred_dicts, recall_dict

