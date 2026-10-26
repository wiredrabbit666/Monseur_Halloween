import math
from dataclasses import dataclass

# COCO keypoint indices used by YOLO pose models
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = range(7)
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


@dataclass
class HeadObservation:
    cx: float  # head center, pixels
    cy: float
    size_px: float  # length of the sizing cue in pixels
    size_m: float  # real-world length of that cue
    cue: str  # which cue was used: "neck", "ears" or "eyes"
    meas_std: float  # noise of log(distance) for this cue


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def head_from_keypoints(xy, conf, cfg):
    """Turn one person's 17 pose keypoints into a head position and size.

    Sizing cues, best first:
      neck - head-keypoint centroid to shoulder midpoint; barely changes
             when the head turns, so it is the most stable
      ears - ear to ear; works when the person faces away
      eyes - eye to eye; small in pixels, so noisiest
    Returns None when there are no usable head keypoints.
    """
    if xy is None or conf is None or len(xy) == 0:
        return None

    vis = conf >= cfg.kp_conf
    head = [i for i in HEAD_POINTS if vis[i]]
    if not head:
        return None
    cx = float(sum(xy[i][0] for i in head) / len(head))
    cy = float(sum(xy[i][1] for i in head) / len(head))

    if vis[L_SHOULDER] and vis[R_SHOULDER]:
        mid = ((xy[L_SHOULDER] + xy[R_SHOULDER]) / 2).tolist()
        size, length, cue, std = _dist((cx, cy), mid), cfg.neck_m, "neck", 0.06
    elif vis[L_EAR] and vis[R_EAR]:
        size, length, cue, std = _dist(xy[L_EAR], xy[R_EAR]), cfg.ear_m, "ears", 0.10
    elif vis[L_EYE] and vis[R_EYE]:
        size, length, cue, std = _dist(xy[L_EYE], xy[R_EYE]), cfg.eye_m, "eyes", 0.15
    else:
        return None

    if not math.isfinite(size) or size < cfg.min_size_px:
        return None
    return HeadObservation(cx, cy, size, length, cue, std)
