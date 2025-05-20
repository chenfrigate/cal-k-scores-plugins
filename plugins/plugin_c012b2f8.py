import numpy as np


def cal_k_scores(prices):
    sorted_prices = np.sort(prices, axis=1)
    n_bidders = prices.shape[1]
    K = np.zeros(prices.shape[0])

    mask_le_3 = n_bidders <= 3
    mask_4_to_6 = (n_bidders >= 4) & (n_bidders <= 6)
    mask_ge_7 = n_bidders >= 7

    K[mask_le_3] = np.mean(sorted_prices[:, :n_bidders], axis=1)[mask_le_3]
    K[mask_4_to_6] = np.mean(sorted_prices[:, : n_bidders - 1], axis=1)[mask_4_to_6]
    K[mask_ge_7] = np.mean(sorted_prices[:, 1 : n_bidders - 1], axis=1)[mask_ge_7]

    deviation = prices / K[:, np.newaxis] - 1
    base_score = 100

    above_0 = deviation > 0
    below_0 = deviation <= 0
    below_minus_02 = deviation <= -0.2
    below_minus_04 = deviation <= -0.4

    scores = base_score - 2 * 100 * deviation * above_0
    scores = np.where(np.isclose(deviation, 0), base_score, scores)
    scores = np.where(below_0, scores + 2 * 100 * deviation, scores)
    scores = np.clip(scores, 50, 150)
    scores = np.where(below_minus_02, scores - 100 * deviation, scores)
    scores = np.where(below_minus_04, 90, scores)

    scores = np.round(scores, 3)
    scores = np.where(scores % 1 >= 0.5, np.ceil(scores), np.floor(scores))

    return scores, K
