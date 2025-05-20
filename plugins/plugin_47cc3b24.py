import numpy as np


def cal_k_scores(prices):
    n_samples, n_bidders = prices.shape
    sorted_prices = np.sort(prices, axis=1)
    valid_counts = np.sum(sorted_prices > 0, axis=1)

    def compute_k(prices, valid_counts):
        n = valid_counts
        k = np.zeros_like(n, dtype=float)
        mask_1 = n <= 3
        mask_2 = (n >= 4) & (n <= 6)
        mask_3 = n >= 7

        k[mask_1] = np.mean(prices[mask_1][:, : n[mask_1]], axis=1)
        k[mask_2] = np.mean(prices[mask_2][:, : n[mask_2] - 1], axis=1)
        k[mask_3] = np.mean(prices[mask_3][:, 1:-1], axis=1)
        return k

    K = compute_k(sorted_prices, valid_counts)
    K_expanded = K[:, np.newaxis]
    price_diffs = (prices - K_expanded) / K_expanded

    score_base = (
        100
        + 200 * np.where(price_diffs < 0, 1, 0)
        - 200 * np.where(price_diffs > 0, 1, 0)
    )
    score_diffs = (
        -2 * price_diffs * 100 * np.where(np.abs(price_diffs) <= 0.2, 1, 0)
        + (-1) * np.where(price_diffs < -0.2, np.abs(price_diffs) - 0.2, 0) * 100
    )
    scores = score_base + score_diffs
    scores = np.clip(scores, 50, 150)
    scores = np.where(np.isclose(price_diffs, -0.4, atol=1e-8), 90, scores)

    scores = np.round(scores * 100) / 100
    return scores, K
