"""Cat eyes that follow the most important person in front of the webcam.

Pupils narrow to a slit as the target's importance rises and open wide
when nobody is in focus; the eyes blink now and then and wander when idle.

Keys: q/Esc quit, f fullscreen, v camera preview, m mirror gaze.
"""

import argparse
import math
import random
import threading
import time

import cv2
import numpy as np

from focus import Camera, FocusConfig, FocusTracker
from focus.overlay import draw_points

WINDOW = "cat eyes"
FUR = (18, 16, 22)
RIM = (6, 6, 8)
SCLERA = (30, 85, 70)
PUPIL = (6, 6, 6)
CORNER = (8, 6, 34)  # shadowed corners of the eye where the eyeball ends
# radial iris colors (BGR): amber around the pupil, yellow-green, dark rim
IRIS_STOPS = [
    (0.00, (20, 150, 235)),
    (0.35, (30, 170, 230)),
    (0.70, (40, 205, 175)),
    (0.92, (30, 140, 90)),
    (1.00, (10, 50, 35)),
]
BLINK_S = 0.18
# iris travel at full gaze, as a fraction of the eye's half-axes; at 1.0 the
# pupil sits on the rim, so ~0.8 puts it right at the edge of the eye
GAZE_RANGE = (0.90, 0.85)
# how far the upper eyelid follows the iris vertically: 0 = lid stays put,
# 1 = lid moves exactly as far as the iris; e.g. 0.5 lowers the lid by half
# of the iris' downward travel (looking up just shows the full eye)
UPPER_LID_FOLLOW = 0.8
# eye opening: lids meet in pointed corners at EYE_WIDTH * rx from the center;
# LID_CURVE < 1 makes the lids fuller, > 1 more pointed
EYE_WIDTH = 1.08
LID_CURVE = 0.8
# open lids' distance above / below center as fractions of ry
EYE_HEIGHT = (0.8, 0.8)
# eyeball half-axes as fractions of (rx, ry); narrower than the opening so a
# dark triangle shows at each corner
EYEBALL = (0.84, 1.05)
# iris radius as a fraction of ry; the pupil slit scales with it
IRIS_SIZE = 1
# pupil half-axes (x, y) relative to the iris radius; x multiplies the slit
# width, which narrows from 0.45 to ~0.08 as focus rises
PUPIL_SIZE = (.9, 0.68)
# pupil outline, pointed at top and bottom: 1 = lens with straight-ish tapering
# sides; < 1 bulges near the tips (gothic arch), > 1 pinches them (spindle)
PUPIL_CURVE = 1.0
SHIFT = 4  # sub-pixel bits for cv2 drawing, keeps slow gaze motion smooth
ONE = 1 << SHIFT


def smooth(current, target, dt, tau):
    """Exponential approach toward target with time constant tau."""
    return current + (target - current) * (1 - math.exp(-dt / tau))


def make_iris(r, seed=7):
    """Pre-render the iris: radial gradient with fine radial streaks."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    rho = np.hypot(xx, yy) / r
    stops = [s for s, _ in IRIS_STOPS]
    img = np.stack(
        [np.interp(rho, stops, [c[ch] for _, c in IRIS_STOPS]) for ch in range(3)], axis=-1
    )
    theta = (np.arctan2(yy, xx) + math.pi) / (2 * math.pi)
    noise = rng.uniform(0.82, 1.12, 240)
    streaks = np.interp(theta * len(noise), np.arange(len(noise) + 1), np.append(noise, noise[0]))
    img *= streaks[..., None]
    alpha = np.clip(r - rho * r + 0.5, 0, 1)[..., None]
    return np.clip(img, 0, 255).astype(np.float32), alpha.astype(np.float32)


def blend(dst, src, alpha, cx, cy):
    """Alpha-blend src (centered at cx, cy) into the uint8 image dst, clipped to bounds."""
    h, w = src.shape[:2]
    x0, y0 = int(round(cx)) - w // 2, int(round(cy)) - h // 2
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if dx1 <= dx0 or dy1 <= dy0:
        return
    s = src[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    a = alpha[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    roi = dst[dy0:dy1, dx0:dx1].astype(np.float32)
    dst[dy0:dy1, dx0:dx1] = (roi * (1 - a) + s * a).astype(np.uint8)


def fp(x, y):
    return int(round(x * ONE)), int(round(y * ONE))


class CatEyes:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.rx = int(width * 0.16)
        self.ry = int(self.rx * 0.82)
        self.iris_r = int(self.ry * IRIS_SIZE)
        self.pad = 12
        self.eyes = [((int(width * 0.3), int(height * 0.5)), 12),
                     ((int(width * 0.7), int(height * 0.5)), -12)]
        self.iris, self.iris_a = make_iris(self.iris_r)

    def render(self, gaze, pupil, openness):
        """gaze: (x, y) in [-1, 1]; pupil: slit half-width as a fraction of
        the iris radius; openness: 1 open, 0 closed."""
        canvas = np.full((self.h, self.w, 3), FUR, np.uint8)
        for center, tilt in self.eyes:
            self._eye(canvas, center, tilt, gaze, pupil, openness)
        return canvas

    def _lids(self, c, tilt, upper, lower):
        """Upper and lower lid curves as fixed-point polylines (corner to corner);
        upper/lower are the lids' signed offsets from c at the middle of the eye."""
        t = np.linspace(-1, 1, 61)
        prof = (1 - t * t) ** LID_CURVE
        x = t * self.rx * EYE_WIDTH
        a = math.radians(tilt)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        curves = []
        for off in (upper, lower):
            pts = np.stack([x, off * prof], 1) @ rot.T + c
            curves.append(np.round(pts * ONE).astype(np.int32))
        return curves

    def _eye(self, canvas, center, tilt, gaze, pupil, openness):
        rx, ry, r = self.rx, self.ry, self.iris_r
        half = int(rx * EYE_WIDTH) + self.pad
        size = 2 * half
        c = (half, half)

        layer = np.full((size, size, 3), SCLERA, np.uint8)
        ix, iy = c[0] + gaze[0] * rx * GAZE_RANGE[0], c[1] + gaze[1] * ry * GAZE_RANGE[1]
        blend(layer, self.iris, self.iris_a, ix, iy)
        pw = max(2.0, r * pupil * PUPIL_SIZE[0])
        t = np.linspace(-1, 1, 61)
        side = pw * (1 - t * t) ** PUPIL_CURVE
        outline = np.concatenate([np.stack([ix + side, iy + t * r * PUPIL_SIZE[1]], 1),
                                  np.stack([ix - side, iy + t * r * PUPIL_SIZE[1]], 1)[::-1]])
        cv2.fillPoly(layer, [np.round(outline * ONE).astype(np.int32)], PUPIL, cv2.LINE_AA, SHIFT)
        # the eyeball ends short of the lid corners, leaving dark triangles
        ball = np.zeros((size, size), np.uint8)
        cv2.ellipse(ball, c, (int(rx * EYEBALL[0]), int(ry * EYEBALL[1])), tilt, 0, 360, 255, -1,
                    cv2.LINE_AA)
        b = (ball.astype(np.float32) / 255)[..., None]
        layer = layer * b + np.array(CORNER, np.float32) * (1 - b)

        # lids meet 0.3*ry below center when closed; the upper lid droops with
        # a downward gaze, scaled by openness so a blink still closes there
        lid_follow = UPPER_LID_FOLLOW * max(iy - c[1], 0.0) * openness
        upper = (0.3 - (0.3 + EYE_HEIGHT[0]) * openness) * ry + lid_follow
        lower = (0.3 + (EYE_HEIGHT[1] - 0.3) * openness) * ry
        up, low = self._lids(c, tilt, upper, lower)
        mask = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mask, [np.concatenate([up, low[::-1]])], 255, cv2.LINE_AA, SHIFT)

        x0, y0 = center[0] - half, center[1] - half
        roi = canvas[y0:y0 + size, x0:x0 + size].astype(np.float32)
        a = (mask.astype(np.float32) / 255)[..., None]
        canvas[y0:y0 + size, x0:x0 + size] = (roi * (1 - a) + layer * a).astype(np.uint8)
        offset = np.array([x0, y0], np.int32) * ONE
        cv2.polylines(canvas, [up + offset, low + offset], False, RIM, max(3, rx // 25),
                      cv2.LINE_AA, SHIFT)


class FocusWorker(threading.Thread):
    """Runs detection on its own thread so the eyes animate at full frame rate."""

    def __init__(self, camera, tracker):
        super().__init__(daemon=True)
        self.camera, self.tracker = camera, tracker
        self._lock = threading.Lock()
        self._state = ([], None, None)
        self.running = True

    def run(self):
        seq = 0
        while self.running:
            try:
                seq, t, frame = self.camera.read(seq)
            except TimeoutError:
                continue
            points = self.tracker.process(frame, t)
            with self._lock:
                self._state = (points, self.tracker.primary, frame)

    def snapshot(self):
        with self._lock:
            return self._state


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--vgain", type=float, default=0.8, help="vertical gaze gain")
    p.add_argument("--no-mirror", action="store_true",
                   help="don't flip horizontally (eyes look toward the person by default)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = FocusConfig(model=args.model, alpha=args.alpha, beta=args.beta)
    camera = Camera(args.camera)
    worker = FocusWorker(camera, FocusTracker(cfg))
    worker.start()
    eyes = CatEyes(args.width, args.height)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, args.width, args.height)

    mirror, show_cam, fullscreen = not args.no_mirror, False, False
    gaze = np.zeros(2)
    pupil = 0.45
    idle_target, next_wander = np.zeros(2), 0.0
    next_blink, blink_start = time.monotonic() + 2.0, None
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt, last = now - last, now
            points, primary, frame = worker.snapshot()
            target_pt = next((p for p in points if p.track_id == primary), None)

            if target_pt is not None:
                x = 1 - target_pt.x if mirror else target_pt.x
                target = np.array([x * 2 - 1, (target_pt.y * 2 - 1) * args.vgain])
                # focusing: the slit narrows as the target's importance rises
                pupil_target = 0.45 - 0.37 * (1 - math.exp(-target_pt.weight))
            else:
                if now >= next_wander:
                    idle_target = np.array([random.uniform(-0.4, 0.4), random.uniform(-0.2, 0.2)])
                    next_wander = now + random.uniform(1.5, 4.0)
                target, pupil_target = idle_target, 0.45
            norm = np.linalg.norm(target)
            if norm > 1:
                target = target / norm
            gaze = smooth(gaze, target, dt, 0.07)
            pupil = smooth(pupil, pupil_target, dt, 0.35)

            openness = 1.0
            if blink_start is None and now >= next_blink:
                blink_start = now
            if blink_start is not None:
                phase = (now - blink_start) / BLINK_S
                if phase >= 1:
                    blink_start, next_blink = None, now + random.uniform(2.5, 7.0)
                else:
                    openness = 1 - math.sin(math.pi * phase)

            img = eyes.render(gaze, pupil, openness)
            if show_cam and frame is not None:
                view = frame.copy()
                draw_points(view, points, primary, labels=False)
                pw = args.width // 4
                ph = int(view.shape[0] * pw / view.shape[1])
                img[-ph - 10:-10, -pw - 10:-10] = cv2.resize(view, (pw, ph))
            cv2.imshow(WINDOW, img)

            spent = time.monotonic() - now
            key = cv2.waitKey(max(1, int((1 / 60 - spent) * 1000))) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("v"):
                show_cam = not show_cam
            if key == ord("m"):
                mirror = not mirror
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        worker.running = False
        worker.join(timeout=2.0)
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
