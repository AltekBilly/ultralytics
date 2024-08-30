# Ultralytics YOLO 🚀, AGPL-3.0 license

# (-/+) -> modfiy by billy
# //from ultralytics.models.yolo import classify, detect, obb, pose, segment, world
from ultralytics.models.yolo import classify, detect, obb, pose, segment, world, altek_landmark
# <- (-/+) modfiy by billy

from .model import YOLO, YOLOWorld

# (-/+) -> modfiy by billy
# //__all__ = "classify", "segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld"
__all__ = "classify", "segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld", "altek_landmark"
# <- (-/+) modfiy by billy