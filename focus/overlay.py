import cv2

PRIMARY = (0, 0, 255)
OTHER = (0, 200, 255)
COASTING = (160, 160, 160)


def draw_points(frame, points, primary_id, labels=True):
    """Draw focus points on a camera frame; circle size scales with weight."""
    h, w = frame.shape[:2]
    max_weight = max((p.weight for p in points), default=1.0)
    for p in points:
        cx, cy = int(p.x * w), int(p.y * h)
        color = COASTING if p.coasting else PRIMARY if p.track_id == primary_id else OTHER
        radius = 12 + int(30 * p.weight / max_weight)
        cv2.circle(frame, (cx, cy), radius, color, 3 if p.track_id == primary_id else 2)
        if labels:
            label = (f"#{p.track_id} w={p.weight:.2f} d={p.distance_m:.2f}m "
                     f"v={p.approach_mps:+.2f}m/s [{p.cue}]")
            cv2.putText(frame, label, (cx - radius, cy - radius - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
