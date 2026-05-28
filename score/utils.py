from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import numpy as np

def compute_eer(labels, scores):
    """sklearn style compute eer
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold


def compute_minDCF(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
    """MinDCF
    Computes the minimum of the detection cost function.  The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def as_norm_score(trials, index_mapping, eval_vectors, cohort_vectors, top_n=200):
    """Compute AS-Norm normalized scores.

    Args:
        trials: Trial list, each row is [label, enroll_path, test_path]
        index_mapping: {path: index_in_eval_vectors}
        eval_vectors: numpy.ndarray, shape = [num_eval_utts, embedding_dim]
        cohort_vectors: numpy.ndarray, shape = [num_cohort_utts, embedding_dim]
        top_n: Select top-N most similar cohort samples

    Returns:
        labels: list[int]
        scores: list[float]  AS-Norm normalized scores
    """
    eval_vectors = np.asarray(eval_vectors, dtype=np.float32)
    cohort_vectors = np.asarray(cohort_vectors, dtype=np.float32)

    # L2 normalization
    def l2_norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return x / n

    eval_norm = l2_norm(eval_vectors)      # [Ne, D]
    cohort_norm = l2_norm(cohort_vectors)  # [Nc, D]

    Ne = eval_norm.shape[0]
    Nc = cohort_norm.shape[0]
    k = min(top_n, Nc)

    # Precompute similarities between eval and cohort: [Ne, Nc]
    sim_eval_cohort = np.matmul(eval_norm, cohort_norm.T)

    # For each eval utterance, compute cohort mean/std
    cohort_mean = np.zeros(Ne, dtype=np.float32)
    cohort_std = np.ones(Ne, dtype=np.float32)

    for i in range(Ne):
        sims = sim_eval_cohort[i]  # [Nc]
        if k <= 0:
            cohort_mean[i] = 0.0
            cohort_std[i] = 1.0
            continue
        topk_idx = np.argpartition(-sims, k - 1)[:k]
        topk_sims = sims[topk_idx]
        m = np.mean(topk_sims)
        s = np.std(topk_sims)
        if s < 1e-6:
            s = 1.0
        cohort_mean[i] = m
        cohort_std[i] = s

    # Precompute raw cosine similarities between eval utterances (for enroll-test raw_score)
    sim_eval_eval = np.matmul(eval_norm, eval_norm.T)  # [Ne, Ne]

    # Compute AS-Norm score per trial
    labels = []
    scores = []

    for item in trials:
        label = int(item[0])
        enroll_idx = index_mapping[item[1]]
        test_idx = index_mapping[item[2]]

        raw_score = sim_eval_eval[enroll_idx, test_idx]

        # Z-Norm using cohort statistics for enroll/test respectively
        z_e = (raw_score - cohort_mean[enroll_idx]) / cohort_std[enroll_idx]
        z_t = (raw_score - cohort_mean[test_idx]) / cohort_std[test_idx]

        as_score = 0.5 * (z_e + z_t)

        labels.append(label)
        scores.append(float(as_score))

    return labels, scores

