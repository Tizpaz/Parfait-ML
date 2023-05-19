import fairlearn

def check_for_fairness(X, y_pred, y_true, a, X_new = None, Y_new = None):
    parities = []
    impacts = []
    eq_odds = []
    metric_frames = []
    metrics = {
        'false positive rate': false_positive_rate,
        'true positive rate': true_positive_rate
    }

    metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=a)
    fair_metric_1 = metric_frame.by_group["true positive rate"]
    fair_metric_2 = metric_frame.by_group["false positive rate"]
    diff_1 = np.abs(fair_metric_1[group_0] - fair_metric_1[group_1])
    diff_2 = np.abs(fair_metric_2[group_0] - fair_metric_2[group_1])
    AOD = (diff_1 + diff_2) * 0.5
    return AOD


constraint_moment = 