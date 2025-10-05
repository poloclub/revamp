from .detectron2_detector import Detectron2Detector
# from detectors.yolov8_detector import Yolov8Detector
from .yolov3_detector import MartYoloDetector
# from detectors.yolov5_detector import Yolov5Detector
# from detectors.yolov11_detector import Yolov11Detector
# from detectors.detr_detector import DetrDetector
# etc.

def load_detector(cfg):
    backend = cfg.scene.detector_name
    if backend == "detectron2":
        return Detectron2Detector(cfg)
    elif backend == "mart_yolo":
        return MartYoloDetector(cfg)
    # elif backend == "yolov5":
    #     return Yolov5Detector(cfg)        
    # elif backend == "yolov8":
    #     return Yolov8Detector(cfg)
    # elif backend == "yolov11":
    #     return Yolov11Detector(cfg)
    # elif backend == "detr":
    #     return DetrDetector(cfg)
    else:
        raise ValueError(f"Unsupported detection backend: {backend}")