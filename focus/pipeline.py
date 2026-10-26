import math
from dataclasses import dataclass

from ultralytics import YOLO

from .head import head_from_keypoints
from .kalman import ConstantVelocityKF
from .scoring import importance


@dataclass
class FocusConfig:
    # model / detection
    model: str = "yolo26n-pose.pt"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    det_conf: float = 0.4
    kp_conf: float = 0.5
    min_size_px: float = 6.0

    # camera geometry: horizontal field of view; refine with calibrate()
    hfov_deg: float = 75.0

    # real-world cue lengths (meters) for an average adult
    neck_m: float = 0.22  # head-keypoint centroid to shoulder midpoint
    ear_m: float = 0.15
    eye_m: float = 0.063

    # filtering
    pos_meas_std: float = 0.01  # normalized image units
    pos_accel_std: float = 3.0
    logd_accel_std: float = 1.5
    reset_jump: float = 0.5  # |delta ln d| above this restarts the filter
    max_coast_s: float = 0.4  # keep predicting a lost track this long

    # scoring knobs
    alpha: float = 1.0  # distance emphasis
    beta: float = 1.0  # approach emphasis
    d_ref: float = 1.0  # meters; proximity = d_ref / d
    proximity_cap: float = 2.0
    rate_cap: float = 2.0  # 1/s
    switch_margin: float = 1.25  # challenger must beat primary by this factor


@dataclass
class FocusPoint:
    track_id: int
    x: float  # head center, 0..1 across the image
    y: float  # 0..1 down the image
    weight: float  # importance
    distance_m: float
    approach_mps: float  # positive = moving toward the camera
    approach_rate: float  # 1/s, = 1 / time-to-contact
    cue: str
    coasting: bool  # not detected this frame, position is predicted


class _Track:
    def __init__(self, x, y, logd, t, obs, cfg):
        self.kx = ConstantVelocityKF(x, cfg.pos_meas_std, cfg.pos_accel_std)
        self.ky = ConstantVelocityKF(y, cfg.pos_meas_std, cfg.pos_accel_std)
        self.kd = ConstantVelocityKF(logd, obs.meas_std, cfg.logd_accel_std)
        self.t = t
        self.last_seen = t
        self.obs = obs

    def predict_to(self, t):
        dt = t - self.t
        for kf in (self.kx, self.ky, self.kd):
            kf.predict(dt)
        self.t = t


class FocusTracker:
    """Webcam frame in, ranked list of FocusPoints out."""

    def __init__(self, cfg=None):
        self.cfg = cfg or FocusConfig()
        self.model = YOLO(self.cfg.model)
        self.tracks = {}
        self.primary_id = None
        self.f_px = None
        self.last_result = None

    def process(self, frame, t):
        cfg = self.cfg
        h, w = frame.shape[:2]
        if self.f_px is None:
            self.f_px = (w / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)

        r = self.model.track(
            frame, persist=True, tracker=cfg.tracker, imgsz=cfg.imgsz,
            conf=cfg.det_conf, verbose=False,
        )[0]
        self.last_result = r

        seen = set()
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().tolist()
            kxy = r.keypoints.xy.cpu().numpy()
            kconf = r.keypoints.conf.cpu().numpy()
            for tid, xy, conf in zip(ids, kxy, kconf):
                obs = head_from_keypoints(xy, conf, cfg)
                if obs is None:
                    continue
                self._observe(tid, obs, w, h, t)
                seen.add(tid)

        points = []
        for tid, tr in list(self.tracks.items()):
            if t - tr.last_seen > cfg.max_coast_s:
                del self.tracks[tid]
                continue
            if tid not in seen:
                tr.predict_to(t)
            points.append(self._to_point(tid, tr, coasting=tid not in seen))

        points.sort(key=lambda p: p.weight, reverse=True)
        self._update_primary(points)
        return points

    @property
    def primary(self):
        return self.primary_id

    def calibrate(self, distance_m):
        """Fix the focal length from the primary person standing at a known
        distance. Returns the new focal length in pixels, or None."""
        tr = self.tracks.get(self.primary_id)
        if tr is None or self.f_px is None:
            return None
        new_f = distance_m * tr.obs.size_px / tr.obs.size_m
        shift = math.log(new_f / self.f_px)
        for other in self.tracks.values():
            other.kd.shift(shift)
        self.f_px = new_f
        return new_f

    def _observe(self, tid, obs, w, h, t):
        x, y = obs.cx / w, obs.cy / h
        logd = math.log(self.f_px * obs.size_m / obs.size_px)
        tr = self.tracks.get(tid)
        if tr is None:
            self.tracks[tid] = _Track(x, y, logd, t, obs, self.cfg)
            return
        tr.predict_to(t)
        if abs(tr.kd.innovation(logd)) > self.cfg.reset_jump:
            # cue switch or ID swap: restart distance rather than fake a lunge
            tr.kd = ConstantVelocityKF(logd, obs.meas_std, self.cfg.logd_accel_std)
        else:
            tr.kd.update(logd, obs.meas_std)
        tr.kx.update(x)
        tr.ky.update(y)
        tr.last_seen = t
        tr.obs = obs

    def _to_point(self, tid, tr, coasting):
        d = math.exp(tr.kd.value)
        rate = -tr.kd.rate
        return FocusPoint(
            track_id=tid,
            x=min(max(tr.kx.value, 0.0), 1.0),
            y=min(max(tr.ky.value, 0.0), 1.0),
            weight=importance(d, rate, self.cfg),
            distance_m=d,
            approach_mps=rate * d,
            approach_rate=rate,
            cue=tr.obs.cue,
            coasting=coasting,
        )

    def _update_primary(self, points):
        if not points:
            self.primary_id = None
            return
        best = points[0]
        current = next((p for p in points if p.track_id == self.primary_id), None)
        if current is None or best.weight > current.weight * self.cfg.switch_margin:
            self.primary_id = best.track_id
