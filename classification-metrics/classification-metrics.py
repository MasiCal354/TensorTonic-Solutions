import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true_arr)

    accuracy = float(np.sum(y_true_arr == y_pred_arr) / n_samples)
    
    def safe_div(num, den):
        return float(num / den) if den > 0 else 0.0

    if average == "binary":
        tp = np.sum((y_true_arr == pos_label) & (y_pred_arr == pos_label))
        fp = np.sum((y_true_arr != pos_label) & (y_pred_arr == pos_label))
        fn = np.sum((y_true_arr == pos_label) & (y_pred_arr != pos_label))
        
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        
    elif average == "micro":
        total_tp = np.sum(y_true_arr == y_pred_arr)
        total_fp = n_samples - total_tp
        total_fn = total_fp
        
        precision = safe_div(total_tp, total_tp + total_fp)
        recall = safe_div(total_tp, total_tp + total_fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        
    else:
        classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        
        precisions = []
        recalls = []
        f1s = []
        weights = []
        
        for c in classes:
            tp = np.sum((y_true_arr == c) & (y_pred_arr == c))
            fp = np.sum((y_true_arr != c) & (y_pred_arr == c))
            fn = np.sum((y_true_arr == c) & (y_pred_arr != c))
            support = np.sum(y_true_arr == c)
            
            p_c = safe_div(tp, tp + fp)
            r_c = safe_div(tp, tp + fn)
            f1_c = safe_div(2 * p_c * r_c, p_c + r_c)
            
            precisions.append(p_c)
            recalls.append(r_c)
            f1s.append(f1_c)
            weights.append(support)
            
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        weights = np.array(weights)
        
        if average == "macro":
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            f1 = float(np.mean(f1s))
        elif average == "weighted":
            total_weight = np.sum(weights)
            precision = safe_div(np.sum(precisions * weights), total_weight)
            recall = safe_div(np.sum(recalls * weights), total_weight)
            f1 = safe_div(np.sum(f1s * weights), total_weight)

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6)
    }
