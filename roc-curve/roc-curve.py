import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    if fps[-1] == 0:
        fpr = np.repeat(0.0, len(fps))
    else:
        fpr = fps / fps[-1]
        
    if tps[-1] == 0:
        tpr = np.repeat(0.0, len(tps))
    else:
        tpr = tps / tps[-1]
        
    thresholds = y_score[threshold_idxs]
    thresholds = np.r_[np.inf, thresholds]
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }
