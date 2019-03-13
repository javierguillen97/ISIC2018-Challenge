# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.iou = []
        self.iou_threshold = []

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            
            iu = np.diag(tmp) / (tmp.sum(axis=1) + tmp.sum(axis=0) - np.diag(tmp))
            self.iou.append(iu[1])
            if iu[1] >= 0.65: self.iou_threshold.append(iu[1])
            else: self.iou_threshold.append(0)
            
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iu = intersect / union
        iu_together = intersect[1:].sum() / union[1:].sum() # concatenate all positive classes, discard background
        
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'IoU': np.nanmean(self.iou),
                'IoU_threshold': np.nanmean(self.iou_threshold),
                'Mean IoU': mean_iu,
                'IoU_all': iu,
                'IoU_together': iu_together}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.iou = []
        self.iou_threshold = []