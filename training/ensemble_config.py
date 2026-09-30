"""Choose the ensemble's voting configuration on the validation set and report on test.

Candidates: Simple Majority, Borda Count and Exponential Voting (alpha from ALPHA_GRID),
each equally weighted and weighted by the members' validation balanced accuracy.
The candidate with the best validation balanced accuracy is kept. The referral
threshold on the malignancy score (vote share of the task's malignant classes) is the
highest one that reaches TARGET_SENSITIVITY for malignant cases on validation.
The test set is used once, for the report.
"""
import numpy as np

import paths  # noqa: F401  (makes common/ importable)
from voting import vote_scores

ALPHA_GRID = [0.01] + [round(0.05 * i, 2) for i in range(1, 20)]  # 0.01, 0.05 ... 0.95
TARGET_SENSITIVITY = 0.95


def balanced_accuracy(pred, y, n_classes):
    present = [c for c in range(n_classes) if np.any(y == c)]
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def choose_configuration(P_val, y_val, n_classes, member_val_bal):
    """Best (method, alpha, weights) on validation balanced accuracy."""
    best = None
    for weights in (None, np.asarray(member_val_bal, dtype=float)):
        for method, alphas in (("majority", [None]), ("borda", [None]), ("exponential", ALPHA_GRID)):
            for a in alphas:
                score = balanced_accuracy(vote_scores(P_val, method, a, weights).argmax(1), y_val, n_classes)
                if best is None or score > best["val_bal_acc"] + 1e-12:
                    best = {"method": method, "alpha": a,
                            "weights": None if weights is None else weights.tolist(), "val_bal_acc": score}
    return best


def referral_threshold(malignancy_val, y_val, malignant_idx, target=TARGET_SENSITIVITY):
    pos = np.sort(malignancy_val[np.isin(y_val, malignant_idx)])
    if len(pos) == 0:
        return 0.5
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def report(scores, y, classes, malignant_idx, threshold):
    """Test metrics of the configured system."""
    pred = scores.argmax(1)
    n = len(classes)
    malignancy = scores[:, malignant_idx].sum(1)
    refer, malignant = malignancy >= threshold, np.isin(y, malignant_idx)
    return {
        "acc": float((pred == y).mean()),
        "bal_acc": balanced_accuracy(pred, y, n),
        "per_class_recall": {c: float((pred[y == i] == i).mean()) if np.any(y == i) else None
                             for i, c in enumerate(classes)},
        "referral_sensitivity": float(refer[malignant].mean()) if malignant.any() else None,
        "referral_specificity": float((~refer[~malignant]).mean()) if (~malignant).any() else None,
        "referral_rate": float(refer.mean()),
    }


def configure(P_val, y_val, P_test, y_test, classes, malignant_idx, member_val_bal):
    cfg = choose_configuration(P_val, y_val, len(classes), member_val_bal)
    s_val = vote_scores(P_val, cfg["method"], cfg["alpha"], cfg["weights"])
    cfg["referral_threshold"] = referral_threshold(s_val[:, malignant_idx].sum(1), y_val, malignant_idx)
    cfg["target_sensitivity"] = TARGET_SENSITIVITY
    s_test = vote_scores(P_test, cfg["method"], cfg["alpha"], cfg["weights"])
    return cfg, report(s_test, y_test, classes, malignant_idx, cfg["referral_threshold"])
