import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def equal_error_rate(y, y_pred):
    """
    Calculate the Equal Error Rate (EER) given the true labels and the scores.

    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.

    Returns:
    float: Equal Error Rate (EER)
    """

    assert np.isnan(y_pred).sum() == 0, "y_pred contains NaNs"
    assert np.isnan(y).sum() == 0, "y contains NaNs"

    if np.unique(y).shape[0] == 1:
        return 0.0

    fpr, tpr, thresholds = roc_curve(y, y_pred)

    fnr = 1 - tpr
    eer_threshold = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer = interp1d(fpr, fnr)(eer_threshold)
    return eer
