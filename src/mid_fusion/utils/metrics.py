import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_det_curve(target_scores, nontarget_scores):
    """
    compute DET curve values

    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores

    output
    ------
      frr:   np.array, FRR, (#N, ), where #N is total number of scores + 1
      far:   np.array, FAR, (#N, ), where #N is total number of scores + 1
      thr:   np.array, threshold, (#N, )
    """

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds, thresholds[min_index]


def equal_error_rate(y, y_pred, implementation="brentq"):
    """
    Calculate the Equal Error Rate (EER) given the true labels and the scores.

    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
    implementation: str, optional (default: 'brentq'), choices : {'brentq', 'asv'}

    Returns:
    float: Equal Error Rate (EER)
    """

    if implementation == "brentq":
        assert np.isnan(y_pred).sum() == 0, "y_pred contains NaNs"
        assert np.isnan(y).sum() == 0, "y contains NaNs"

        if np.unique(y).shape[0] == 1:
            return 0.0

        fpr, tpr, thresholds = roc_curve(y, y_pred)

        fnr = 1 - tpr
        eer_threshold = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer = interp1d(fpr, fnr)(eer_threshold)
    elif implementation == "asv":
        eer, _, _, _, _ = compute_eer(y[y_pred == 1], y[y_pred == 0])
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    return eer
