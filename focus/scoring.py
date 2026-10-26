import math


def importance(distance_m, approach_rate, cfg):
    """Importance = proximity^alpha * exp(beta * approach_rate).

    proximity     = d_ref / distance, capped at proximity_cap
    approach_rate = -d(ln distance)/dt in 1/s (= 1 / time-to-contact);
                    positive when approaching, negative when receding
    alpha scales the weight of distance, beta the weight of motion.
    """
    if not math.isfinite(distance_m) or not math.isfinite(approach_rate):
        return 0.0
    proximity = min(cfg.d_ref / max(distance_m, 1e-3), cfg.proximity_cap)
    rate = max(-cfg.rate_cap, min(cfg.rate_cap, approach_rate))
    return proximity**cfg.alpha * math.exp(cfg.beta * rate)
