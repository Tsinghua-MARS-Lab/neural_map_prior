import numpy as np
from scipy.spatial import distance
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def vec_iou(pred_lines, gt_lines):
    '''
        each line with 1 meter width
        pred_lines: num_preds, npts, 2
        gt_lines: num_gts, npts, 2
    '''

    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]

    pred_lines_shapely = \
        [LineString(i).buffer(1.,
                              cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
         for i in pred_lines]
    gt_lines_shapely = \
        [LineString(i).buffer(1.,
                              cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
         for i in gt_lines]

    # construct tree
    tree = STRtree(gt_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(gt_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(pred_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                gt_id = index_by_id[id(o)]

                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[i, gt_id] = inter / union

    return iou_matrix


def convex_iou(pred_lines, gt_lines, gt_mask):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''

    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    pred_lines_shapely = \
        [Polygon(i).convex_hull for i in pred_lines]
    gt_lines_shapely = \
        [Polygon(i[m].reshape(-1, 2)).convex_hull for i, m in zip(gt_lines, gt_mask)]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(gt_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[pred_id, i] = inter / union

    return iou_matrix


def rbbox_iou(pred_lines, gt_lines, gt_mask):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''

    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    pred_lines_shapely = \
        [Polygon(i).minimum_rotated_rectangle for i in pred_lines]
    gt_lines_shapely = \
        [Polygon(i[m].reshape(-1, 2)) for i, m in zip(gt_lines, gt_mask)]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(gt_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[pred_id, i] = inter / union

    return iou_matrix


def polyline_score(pred_lines, gt_lines, linewidth=1.):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''
    positive_threshold = 1.
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = \
        [LineString(i).buffer(linewidth,
                              cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
         for i in pred_lines]
    gt_lines_shapely = \
        [LineString(i).buffer(linewidth,
                              cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
         for i in gt_lines]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(gt_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                dist_mat = distance.cdist(
                    pred_lines[pred_id], gt_lines[i], 'euclidean')

                valid_ab = (dist_mat.min(-1) < positive_threshold).sum()
                valid_ba = (dist_mat.min(-2) < positive_threshold).sum()

                # iou_matrix[pred_id, i] = ((valid_ba+valid_ab)/2) / line_length
                # import ipdb; ipdb.set_trace()
                # if True:
                #     import matplotlib.pyplot as plt
                #     plt.plot(pred_lines[pred_id][:,0],pred_lines[pred_id][:,1],'-',color='red',alpha=0.5)
                #     plt.plot(gt_lines[i][:,0],gt_lines[i][:,1],'-',color='blue',alpha=0.5)
                #     plt.savefig('test.png')
                #     plt.close()
                iou_matrix[pred_id, i] = min(valid_ba, valid_ab) / line_length
                # if  not (iou_matrix[pred_id, i] <= 1. and iou_matrix[pred_id, i] >= 0.):
                #     import ipdb; ipdb.set_trace()
                assert iou_matrix[pred_id, i] <= 1. and iou_matrix[pred_id, i] >= 0.

    return iou_matrix


if __name__ == '__main__':
    import torch

    line1 = torch.tensor([
        [1, 4], [3, 5], [5, 5]
    ])

    line0 = torch.tensor([
        [3, 6], [4, 6], [5, 8]
    ])

    line2 = torch.tensor([
        [1, 3], [3, 4], [5, 4]
    ])

    line3 = torch.tensor([
        [4, 1], [3, 3], [5, 3]
    ])

    gt = torch.stack((line2, line3), dim=0).type(torch.float32)
    pred = torch.stack((line0, line1), dim=0).type(torch.float32)

    # import ipdb; ipdb.set_trace()
    # with mmcv.Timer():
    #     gt = upsampler(gt, pts=10)
    #     pred = upsampler(pred, pts=10)

    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    from descartes import PolygonPatch

    iou_matrix = vec_iou(pred, gt)
    print(iou_matrix)

    fig, ax = plt.subplots()
    for i in gt:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], 'o', color='red')
        plt.plot(i[:, 0], i[:, 1], '-', color='red')

        dilated = LineString(i).buffer(1, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
        patch1 = PolygonPatch(dilated, fc='red', ec='red', alpha=0.5, zorder=-1)
        ax.add_patch(patch1)

    for i in pred:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], 'o', color='blue')
        plt.plot(i[:, 0], i[:, 1], '-', color='blue')

        dilated = LineString(i).buffer(1, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
        patch1 = PolygonPatch(dilated, fc='blue', ec='blue', alpha=0.5, zorder=-1)
        ax.add_patch(patch1)

    ax.axis('equal')

    plt.savefig('test.png')
