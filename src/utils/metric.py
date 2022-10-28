import numpy as np
from numba import jit
from sklearn.metrics import average_precision_score, balanced_accuracy_score


def get_metrics(probs, preds, y):
    samples = float(len(y))
    tn, fp, fn, tp = confusion_matrix_custom(y, preds)
    tnr, fpr, fnr, tpr = tn / (tn + fp), fp / (fp + tn), fn / (tp + fn), tp / (tp + fn)

    precision = precision_score(y, preds)
    auc = fast_auc(y, probs)

    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    aupr = average_precision_score(y, preds)

    fp_idx = np.logical_and(y == 0, preds == 1)
    pos_idx = y == 1

    acc = float(np.sum(y == preds) / samples)

    fp_conf = np.mean(probs[fp_idx])
    pos_conf = np.mean(probs[pos_idx])

    fp_count = int(np.sum(fp_idx))
    total_samples = len(y)

    youden = balanced_accuracy_score(y, preds)

    rates = {"tnr": tnr, "fpr": fpr, "fnr": fnr, "tpr": tpr, "precision": precision, "recall": recall, "f1": f1,
             "auc": auc, "aupr": aupr, "loss": None, "fp_conf": fp_conf, "pos_conf": pos_conf, "fp_count": fp_count,
             "total_samples": total_samples, "acc": acc, "detection": None, "youden": youden}

    return rates


@jit
def fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == np.sum(y_true):
        return 0.0
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))
    return auc


def confusion_matrix_custom(y: np.ndarray, y_pred: np.ndarray):
    tn = int(np.sum(np.logical_and(y_pred == 0, y == 0)))
    tp = int(np.sum(np.logical_and(y_pred == 1, y == 1)))

    fp = int(np.sum(np.logical_and(y_pred == 1, y == 0)))
    fn = int(np.sum(np.logical_and(y_pred == 0, y == 1)))

    return tn, fp, fn, tp


def precision_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def recall_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    return tp / (tp + fn)


def f1_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except:
        f1 = 0

    return f1
