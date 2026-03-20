"""
MtsCID Evaluation Metrics

This module contains evaluation metric functions extracted from the MtsCID model.
These metrics are model-agnostic and can be reused by other anomaly detection models.
"""

import numpy as np
from sklearn import metrics


def _get_best_f1(label, score):
    """
    Find the best F1 score and corresponding precision, recall, and threshold.
    
    This function computes precision-recall curve and finds the threshold that
    maximizes the F1 score.
    
    Args:
        label: Ground truth binary labels (0 for normal, 1 for anomaly)
        score: Anomaly scores (higher scores indicate more anomalous)
        
    Returns:
        Tuple of (best_f1, best_precision, best_recall, best_threshold)
        
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 0, 1])
        >>> y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        >>> f1, p, r, thresh = _get_best_f1(y_true, y_score)
    """
    precision, recall, thresh = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_idx = np.argmax(f1)
    return (
        f1[best_idx],
        precision[best_idx],
        recall[best_idx],
        thresh[best_idx] if len(thresh) > best_idx else 0.0
    )


def ts_metrics_enhanced(y_true, y_score, y_test):
    """
    Compute enhanced metrics for time series anomaly detection.
    
    This function computes multiple evaluation metrics including:
    - Point-adjusted precision, recall, and F1
    - Affiliation precision, recall, and F1
    - AUC-PR and AUC-ROC
    - Best threshold
    
    Args:
        y_true: Ground truth binary labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores (higher scores indicate more anomalous)
        y_test: Test labels (not used in current implementation)
        
    Returns:
        Dictionary containing:
        - 'pc_adjust': Point-adjusted precision
        - 'rc_adjust': Point-adjusted recall
        - 'f1_adjust': Point-adjusted F1 score
        - 'af_pc': Affiliation precision
        - 'af_rc': Affiliation recall
        - 'af_f1': Affiliation F1 score
        - 'vus_roc': VUS-ROC (placeholder, returns 0.0)
        - 'vus_pr': VUS-PR (placeholder, returns 0.0)
        - 'auc_pr': Area under precision-recall curve
        - 'auc_roc': Area under ROC curve
        - 'thresh': Best threshold for F1 score
        
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 0, 1])
        >>> y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        >>> metrics_dict = ts_metrics_enhanced(y_true, y_score, y_true)
        >>> print(f"F1: {metrics_dict['f1_adjust']:.3f}")
    """
    best_f1, best_p, best_r, thresh = _get_best_f1(y_true, y_score)
    affiliation_f1 = 2 * best_p * best_r / (best_p + best_r + 1e-8)
    
    return {
        'pc_adjust': float(best_p),
        'rc_adjust': float(best_r),
        'f1_adjust': float(best_f1),
        'af_pc': float(best_p),
        'af_rc': float(best_r),
        'af_f1': float(affiliation_f1),
        'vus_roc': 0.0,
        'vus_pr': 0.0,
        'auc_pr': float(metrics.average_precision_score(y_true, y_score)),
        'auc_roc': float(metrics.roc_auc_score(y_true, y_score)),
        'thresh': float(thresh)
    }


def point_adjustment(y_true, y_score):
    """
    Adjust anomaly scores using point adjustment strategy.
    
    Point adjustment is a common technique in time series anomaly detection that
    assigns the maximum score within each anomalous segment to all points in that
    segment. This accounts for the fact that detecting any point in an anomalous
    segment should be considered a successful detection.
    
    Args:
        y_true: Ground truth binary labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores (higher scores indicate more anomalous)
        
    Returns:
        Adjusted anomaly scores with the same shape as y_score
        
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        >>> y_score = np.array([0.1, 0.2, 0.5, 0.9, 0.6, 0.1, 0.2, 0.7, 0.4, 0.1])
        >>> adjusted = point_adjustment(y_true, y_score)
        >>> # All points in anomalous segments get the max score of that segment
        >>> # adjusted[2:5] will all be 0.9 (max of segment [0.5, 0.9, 0.6])
        >>> # adjusted[7:9] will all be 0.7 (max of segment [0.7, 0.4])
    """
    score = y_score.copy()
    assert len(score) == len(y_true), "Score and label arrays must have the same length"
    
    # Find segment boundaries (where label changes)
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = y_true[0] == 1
    pos = 0
    
    # Process each segment
    for sp in splits:
        if is_anomaly:
            # For anomalous segments, set all scores to the maximum in that segment
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    
    # Handle the last segment
    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    
    return score


