# import pandas as pd
import numpy as np
import torch
# import torch.nn.functional as F
# from sklearn.metrics import hamming_loss as hl
# from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_loss
# from sklearn.metrics import accuracy_score


def average_precision(Outputs, target):
    # remove samples with all 0 or 1 targets
    non_all_zeros = ((target == 1).sum(dim=1) > 0) & ((target == 0).sum(dim=1) < target.size(1))
    Outputs = Outputs[non_all_zeros]
    target = target[non_all_zeros]

    # compute average precision for each sample
    avg_precisions = []
    for i in range(Outputs.size(0)):
        # get targets and outputs for this sample
        sample_target = target[i]
        sample_output = Outputs[i]

        # sort predictions by descending order of confidence
        sorted_output, sorted_indices = torch.sort(sample_output, descending=True)
        sample_target = sample_target[sorted_indices]

        # compute precision at each threshold
        tp = sample_target.cumsum(dim=0)
        fp = (1-sample_target).cumsum(dim=0)
        precision = tp.float() / (tp + fp).float()

        # compute average precision
        recall = tp.float() / sample_target.sum().float()
        ap = ((precision * sample_target.float()).sum() / sample_target.sum().float()).item()
        avg_precisions.append(ap)

    # compute mean of average precisions over all samples
    mean_avg_precision = torch.tensor(avg_precisions).mean().item()

    return mean_avg_precision


def accuracy(Y_hat, Y):

    Y[Y != 1] = 0
    Y_hat[Y_hat != 1] = 0

    num_samples = Y.shape[0]
    result = 0

    for i in range(num_samples):
        Y_i = Y[i, :]
        Y_hat_i = Y_hat[i, :]


        intersect = np.logical_and(Y_i, Y_hat_i)
        union = np.logical_or(Y_i, Y_hat_i)

        union_nonzero_count = np.count_nonzero(union)
        if union_nonzero_count == 0:
            result_i = 1
        else:
            result_i = np.count_nonzero(intersect) / union_nonzero_count

        result += result_i

    result = result / num_samples
    return result



def ranking_loss(Outputs, target):
    RankingLoss = label_ranking_loss(target, Outputs)
    return RankingLoss



