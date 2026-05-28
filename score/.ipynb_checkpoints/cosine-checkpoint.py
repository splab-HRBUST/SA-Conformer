import numpy as np
import logging
import os

logger = logging.getLogger('libs')


def cosine_score(trials, index_mapping, eval_vectors):
    labels = []
    scores = []

    # 创建反向映射：路径 -> 索引
    path_to_index = {path: idx for idx, path in enumerate(index_mapping.keys())}

    missing_count = 0
    for item in trials:
        if len(item) < 3:
            continue

        enroll_path = item[1]
        test_path = item[2]

        # 检查路径是否存在映射中
        if enroll_path not in path_to_index:
            # 尝试使用 basename 匹配
            enroll_basename = os.path.basename(enroll_path)
            found = False
            for full_path, idx in path_to_index.items():
                if enroll_basename == os.path.basename(full_path):
                    enroll_path = full_path
                    found = True
                    break
            if not found:
                missing_count += 1
                continue

        if test_path not in path_to_index:
            # 尝试使用 basename 匹配
            test_basename = os.path.basename(test_path)
            found = False
            for full_path, idx in path_to_index.items():
                if test_basename == os.path.basename(full_path):
                    test_path = full_path
                    found = True
                    break
            if not found:
                missing_count += 1
                continue

        enroll_idx = path_to_index[enroll_path]
        test_idx = path_to_index[test_path]

        enroll_vector = eval_vectors[enroll_idx]
        test_vector = eval_vectors[test_idx]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)

        # 避免除零错误
        if denom == 0:
            score = 0.0
        else:
            score = score / denom

        labels.append(int(item[0]))
        scores.append(score)

    if missing_count > 0:
        logger.warning(f"Missing {missing_count} trial pairs due to path mapping issues")

    return labels, scores


def s_norm_score(trials, index_mapping, eval_vectors, cohort_vectors, cohort_mapping):
    """
    对称归一化 (S-Norm)
    """
    logger.info("Applying S-Norm...")

    # 归一化embedding
    eval_vectors_norm = eval_vectors / np.linalg.norm(eval_vectors, axis=1, keepdims=True)
    cohort_vectors_norm = cohort_vectors / np.linalg.norm(cohort_vectors, axis=1, keepdims=True)

    # 计算所有eval vectors与cohort vectors的得分
    logger.info("Computing cohort scores...")
    cohort_scores = np.dot(eval_vectors_norm, cohort_vectors_norm.T)

    # 计算每个eval vector的均值和标准差
    logger.info("Computing means and stds...")
    eval_means = np.mean(cohort_scores, axis=1)
    eval_stds = np.std(cohort_scores, axis=1)

    # 创建从路径到索引的映射
    path_to_index = {path: idx for idx, path in enumerate(index_mapping.keys())}

    labels = []
    snorm_scores = []
    missing_count = 0

    logger.info("Processing trials with S-Norm...")
    for trial in trials:
        if len(trial) < 3:
            continue

        enroll_path, test_path, label = trial[1], trial[2], trial[0]

        # 检查路径是否存在映射中
        if enroll_path not in path_to_index or test_path not in path_to_index:
            missing_count += 1
            continue

        enroll_idx = path_to_index[enroll_path]
        test_idx = path_to_index[test_path]

        # 原始cosine得分
        original_score = np.dot(eval_vectors_norm[enroll_idx], eval_vectors_norm[test_idx])

        # S-Norm归一化
        enroll_mean = eval_means[enroll_idx]
        enroll_std = eval_stds[enroll_idx] if eval_stds[enroll_idx] > 1e-8 else 1.0
        test_mean = eval_means[test_idx]
        test_std = eval_stds[test_idx] if eval_stds[test_idx] > 1e-8 else 1.0

        # 对称归一化
        snorm_score = 0.5 * (
                (original_score - enroll_mean) / enroll_std +
                (original_score - test_mean) / test_std
        )

        labels.append(int(label))
        snorm_scores.append(snorm_score)

    if missing_count > 0:
        logger.warning(f"S-Norm: Missing {missing_count} trial pairs due to path mapping issues")

    logger.info(f"S-Norm processed {len(labels)} trials")
    return labels, snorm_scores


def as_norm_score(trials, index_mapping, eval_vectors, cohort_vectors, cohort_mapping, top_n=300):
    """
    自适应对称归一化 (AS-Norm)
    """
    logger.info(f"Applying AS-Norm with top_n={top_n}...")

    # 归一化embedding
    eval_vectors_norm = eval_vectors / np.linalg.norm(eval_vectors, axis=1, keepdims=True)
    cohort_vectors_norm = cohort_vectors / np.linalg.norm(cohort_vectors, axis=1, keepdims=True)

    # 创建从路径到索引的映射
    path_to_index = {path: idx for idx, path in enumerate(index_mapping.keys())}

    labels = []
    asnorm_scores = []
    missing_count = 0

    logger.info("Processing trials with AS-Norm...")
    for trial_idx, trial in enumerate(trials):
        if len(trial) < 3:
            continue

        enroll_path, test_path, label = trial[1], trial[2], trial[0]

        # 检查路径是否存在映射中
        if enroll_path not in path_to_index or test_path not in path_to_index:
            missing_count += 1
            continue

        enroll_idx = path_to_index[enroll_path]
        test_idx = path_to_index[test_path]

        # 原始cosine得分
        original_score = np.dot(eval_vectors_norm[enroll_idx], eval_vectors_norm[test_idx])

        # 为当前enroll选择top_n cohort scores
        enroll_cohort_scores = np.dot(eval_vectors_norm[enroll_idx:enroll_idx + 1], cohort_vectors_norm.T)[0]
        top_enroll_indices = np.argsort(enroll_cohort_scores)[-top_n:]
        enroll_mean = np.mean(enroll_cohort_scores[top_enroll_indices])
        enroll_std = np.std(enroll_cohort_scores[top_enroll_indices]) if len(top_enroll_indices) > 1 else 1.0

        # 为当前test选择top_n cohort scores
        test_cohort_scores = np.dot(eval_vectors_norm[test_idx:test_idx + 1], cohort_vectors_norm.T)[0]
        top_test_indices = np.argsort(test_cohort_scores)[-top_n:]
        test_mean = np.mean(test_cohort_scores[top_test_indices])
        test_std = np.std(test_cohort_scores[top_test_indices]) if len(top_test_indices) > 1 else 1.0

        # 自适应对称归一化
        asnorm_score = 0.5 * (
                (original_score - enroll_mean) / enroll_std +
                (original_score - test_mean) / test_std
        )

        labels.append(int(label))
        asnorm_scores.append(asnorm_score)

        if trial_idx % 1000 == 0:
            logger.info(f"Processed {trial_idx} trials")

    if missing_count > 0:
        logger.warning(f"AS-Norm: Missing {missing_count} trial pairs due to path mapping issues")

    logger.info(f"AS-Norm processed {len(labels)} trials")
    return labels, asnorm_scores