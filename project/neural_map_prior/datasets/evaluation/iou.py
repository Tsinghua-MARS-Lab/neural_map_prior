import torch


def get_batch_iou(pred_map, gt_map, device):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects).to(device), torch.tensor(unions).to(device)


def get_batch_iou_bound(pred_map, gt_map, device):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float() - (in_bound_with_dis(pred) & in_bound_with_dis(tgt)).sum().float()
            union = (pred | tgt).sum().float() - (in_bound_with_dis(pred) | in_bound_with_dis(tgt)).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects).to(device), torch.tensor(unions).to(device)


def in_bound_with_dis(map):
    area_ratio = 5
    part_level = 2
    H = 200
    # W = 400
    # W = 200
    W = 320
    B, cur_H, cur_W = map.shape
    part_h = (H // area_ratio) * part_level
    part_w = (W // area_ratio) * part_level
    new_map = map[:, cur_H // 2 - part_h // 2: cur_H // 2 + part_h // 2,
              cur_W // 2 - part_w // 2: cur_W // 2 + part_w // 2]
    print('new_map', new_map.shape)
    return new_map


def get_batched_iou_no_reduction(pred_map, gt_map, device):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum((-1, -2)).float()
            union = (pred | tgt).sum((-1, -2)).float()
            intersects.append(intersect)
            unions.append(union)

    intersects = torch.stack(intersects).to(device)
    unions = torch.stack(unions).to(device)
    unions_mask = unions > 0
    return (intersects * unions_mask) / unions.masked_fill(~unions_mask, 1)
