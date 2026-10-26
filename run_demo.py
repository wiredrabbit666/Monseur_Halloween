"""Live webcam demo: shows focus points and importance weights.

Keys: q/Esc quit, c calibrate (stand at --calib-distance from the camera),
s toggle skeleton, p toggle printing the focus list.
Trackbars adjust alpha (distance emphasis) and beta (approach emphasis).
"""

import argparse
import time

import cv2

from focus import Camera, FocusConfig, FocusTracker
from focus.overlay import draw_points

WINDOW = "focus"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--hfov", type=float, default=75.0, help="horizontal field of view, degrees")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--calib-distance", type=float, default=1.0, help="meters, used by the c key")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = FocusConfig(model=args.model, imgsz=args.imgsz, hfov_deg=args.hfov,
                      alpha=args.alpha, beta=args.beta)
    tracker = FocusTracker(cfg)
    cam = Camera(args.camera)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("alpha x100", WINDOW, int(cfg.alpha * 100), 300, lambda v: None)
    cv2.createTrackbar("beta x100", WINDOW, int(cfg.beta * 100), 300, lambda v: None)

    show_skeleton, print_points = False, False
    seq, fps, last = 0, 0.0, time.monotonic()
    try:
        while True:
            seq, t, frame = cam.read(seq)
            cfg.alpha = cv2.getTrackbarPos("alpha x100", WINDOW) / 100
            cfg.beta = cv2.getTrackbarPos("beta x100", WINDOW) / 100

            points = tracker.process(frame, t)

            view = tracker.last_result.plot(boxes=False) if show_skeleton else frame.copy()
            draw_points(view, points, tracker.primary)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(view, f"{fps:4.1f} fps  alpha={cfg.alpha:.2f} beta={cfg.beta:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)

            if print_points and points:
                print([(p.track_id, round(p.x, 3), round(p.y, 3), round(p.weight, 3))
                       for p in points])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                show_skeleton = not show_skeleton
            if key == ord("p"):
                print_points = not print_points
            if key == ord("c"):
                f = tracker.calibrate(args.calib_distance)
                print(f"calibrated focal length: {f:.0f}px" if f else "calibration needs one visible person")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
