import numpy as np
def cal_k_scores(prices):
    n = len(prices)
    if n <= 3:
        K = np.mean(prices)
    elif 4 <= n <= 6:
        K = np.mean(np.sort(prices)[:-1])
    else:  # n >= 7
        K = np.mean(np.sort(prices)[1:-1])
    
    scores = np.zeros_like(prices, dtype=float)
    for i, P in enumerate(prices):
        if P == K:
            scores[i] = 100.0
        else:
            ratio = (P - K) / K * 100  # percentage difference
            if ratio > 0:  # higher than K
                deduction = ratio * 2
                score = 100 - deduction
                scores[i] = max(score, 50.0)
            else:  # lower than K
                ratio_abs = abs(ratio)
                if ratio_abs <= 20:
                    addition = ratio_abs * 2
                    score = 100 + addition
                    scores[i] = min(score, 150.0)
                elif 20 < ratio_abs < 40:
                    deduction = (ratio_abs - 20) * 1
                    score = 150 - deduction
                    scores[i] = max(score, 90.0)
                else:  # ratio_abs >= 40
                    scores[i] = 90.0
    
    # Round to 2 decimal places
    scores = np.round(scores, 2)
    return scores, float(K)