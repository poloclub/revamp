from typing import List, Dict, Any, Optional
import os
import torch as ch
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
from types import SimpleNamespace
from urllib.request import urlretrieve
from .base_detector import BaseDetector
import yaml
from mart.models.detection.yolo import yolo_darknet as mart_yolo_darknet, yolov4 as mart_yolov4

# COCO-80 canonical class names (Darknet/Ultralytics order)
COCO80_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
    'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

class MartYoloDetector(BaseDetector):
    """
    Detector wrapper that uses MART's native YOLO pipeline end-to-end.
    Supports two backends:
      - 'darknet'  : uses mart.models.detection.yolo.yolo_darknet(cfg, weights, ...)
      - 'yolov4'   : uses mart.models.detection.yolo.yolov4(...). (No pretrained weights by default.)

    Expects the same external interface as Yolov3Detector:
      - load_model()
      - infer(x, target, bboxes, batch_size=1) -> differentiable loss tensor
      - predict_and_save(image, path, ...) -> writes visualization; optionally returns a small dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.class_names: Optional[List[str]] = None
        self._warned_names_mismatch = False
        # Keep consistent with the rest of the pipeline
        self.expects_unit_input = True
        # Debug toggle
        try:
            self.debug = bool(getattr(cfg.model, "debug_yolo", False))
        except Exception:
            self.debug = False

        # Inference thresholds (can be overridden per-call)
        self.confidence_threshold = float(getattr(getattr(cfg, "model", object()), "mart_conf_threshold", 0.25))
        self.nms_threshold = float(getattr(getattr(cfg, "model", object()), "mart_nms_threshold", 0.45))
        self.max_det = int(getattr(getattr(cfg, "model", object()), "mart_max_det", 300))

    # ------------------------- utils -------------------------
    def _select_device(self) -> str:
        try:
            return f"cuda:{self.cfg.device}"
        except Exception:
            return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _ensure_names(self, nc: int) -> None:
        # Prefer names supplied in cfg; otherwise create generic names
        names = None
        try:
            names = getattr(getattr(self.cfg, "data", object()), "names", None)
        except Exception:
            names = None
        if names is None:
            try:
                names = getattr(getattr(self.cfg, "model", object()), "class_names", None)
            except Exception:
                names = None
        if isinstance(names, dict):
            # Normalize dict to ordered list by numeric key if possible
            try:
                names = [names[k] for k in sorted(names.keys(), key=lambda k: int(k))]
            except Exception:
                names = list(names.values())
        if isinstance(names, (list, tuple)) and len(names) == nc:
            self.class_names = [str(x) for x in names]
            return
        # No valid names provided: use COCO if 80 classes, else generic
        if int(nc) == 80:
            self.class_names = COCO80_NAMES.copy()
        else:
            self.class_names = [f"class_{i}" for i in range(nc)]
    
    def _download_darknet_defaults(self) -> (str, str):
        """
        Downloads YOLOv4-tiny cfg + **detector** weights into ./pretrained-models/mart_darknet
        if paths are not provided. Returns (cfg_path, weights_path).
        """
        root = os.path.join("pretrained-models", "mart_darknet")
        os.makedirs(root, exist_ok=True)

        # Canonical (2-head) tiny – widely available and fine for viz
        cfg_path = os.path.join(root, "yolov4-tiny.cfg")
        weights_path = os.path.join(root, "yolov4-tiny.weights")

        if not os.path.exists(cfg_path):
            urlretrieve(
                "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
                cfg_path,
            )
        if not os.path.exists(weights_path):
            urlretrieve(
                "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
                weights_path,
            )
        return cfg_path, weights_path    

    def _build_targets(self, x: torch.Tensor, target: torch.Tensor, bboxes: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Convert pipeline's (target, bboxes) into MART-format targets:
        - List[dict(boxes: Float[N_i,4], labels: Long[N_i])]

        Accepts bboxes shaped [N, 4] (one box per image), [N, K, 4], or [1, K, 4].
        Boxes must be in (x1,y1,x2,y2) image-space pixels (same coordinate frame as x[i]).
        """
        device = x.device
        N = x.shape[0] if x.dim() == 4 else 1

        # Ensure tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        bboxes = bboxes.to(device=device, dtype=torch.float32)

        # Normalize bbox shape
        if bboxes.dim() == 1:
            # single box for a single image
            assert N == 1, f"Got 1D bboxes for batch N={N}; ambiguous."
            bboxes = bboxes.view(1, 1, 4)
        elif bboxes.dim() == 2:
            if bboxes.shape == (N, 4):
                # one box per image
                bboxes = bboxes.view(N, 1, 4)
            elif N == 1 and bboxes.shape[-1] == 4:
                # single-image, multiple boxes
                bboxes = bboxes.view(1, -1, 4)
            else:
                raise RuntimeError(f"Expected bboxes with shape [N,4] or [1,K,4], got {tuple(bboxes.shape)} for N={N}")
        elif bboxes.dim() == 3:
            if bboxes.shape[-1] != 4:
                raise RuntimeError(f"Last bbox dim must be 4, got {bboxes.shape[-1]}")
            if bboxes.shape[0] not in (1, N):
                raise RuntimeError(f"First bbox dim must be 1 or N={N}, got {bboxes.shape[0]}")
        else:
            raise RuntimeError(f"Expected bboxes with shape [N,K,4] or [N,4], got {tuple(bboxes.shape)}")

        # Normalize/prepare targets
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        target = target.to(device=device, dtype=torch.long)
        if target.dim() == 0:
            target = target.view(1).repeat(N)
        elif target.shape[0] == 1 and N > 1:
            target = target.repeat(N)
        elif target.shape[0] != N:
            raise RuntimeError(f"Target length ({target.shape[0]}) must be 1 or equal to batch size ({N}).")

        targets: List[Dict[str, torch.Tensor]] = []
        for i in range(N):
            boxes_i = bboxes[i]
            if boxes_i.numel() == 0:
                targets.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=device),
                })
                continue
            if boxes_i.dim() == 1:
                boxes_i = boxes_i.view(1, 4)
            labels_i = torch.full((boxes_i.shape[0],), int(target[i].item()), dtype=torch.int64, device=device)
            targets.append({
                "boxes": boxes_i.to(device=device, dtype=torch.float32),
                "labels": labels_i,
            })
        return targets

    # ------------------------- public API -------------------------
    def load_model(self):
        """
        Instantiates a MART YOLO model (either from Darknet cfg/weights or torchvision-style YOLOv4)
        and moves it to the configured device.

        Config options (all optional):
          cfg.model.mart_backend         : 'darknet' (default) or 'yolov4'
          cfg.model.cfg_path             : path to Darknet .cfg (darknet backend)
          cfg.model.weights_path         : path to Darknet .weights/.conv.* (darknet backend)
          cfg.model.auto_download        : bool, download default yolov4-tiny if paths missing (default True)
          cfg.model.num_classes          : int (yolov4 backend only; darknet infers from cfg)
          cfg.model.mart_conf_threshold  : float (default 0.25)
          cfg.model.mart_nms_threshold   : float (default 0.45)
          cfg.model.mart_max_det         : int (default 300)
        """
        logger = logging.getLogger("mart_yolo")
        device_str = self._select_device()

        backend = str(getattr(getattr(self.cfg, "model", object()), "mart_backend", "darknet")).lower()
        auto_download = bool(getattr(getattr(self.cfg, "model", object()), "auto_download", True))

        if backend == "darknet":
            cfg_path = getattr(getattr(self.cfg, "model", object()), "cfg_path", None)
            weights_path = getattr(getattr(self.cfg, "model", object()), "weights_path", None)

            # Emit a helpful message before trying to fetch defaults
            if (not cfg_path or not os.path.exists(str(cfg_path))) or (not weights_path or not os.path.exists(str(weights_path))):
                if auto_download:
                    logger.info("MART Darknet paths missing; downloading yolov4-tiny defaults...")
                    try:
                        cfg_path, weights_path = self._download_darknet_defaults()
                    except Exception as e:
                        raise RuntimeError(
                            f"Auto-download of Darknet defaults failed ({e}). "
                            f"Provide cfg.model.cfg_path and cfg.model.weights_path."
                        )
                else:
                    raise FileNotFoundError(
                        "cfg.model.cfg_path / cfg.model.weights_path not found and auto_download is False."
                    )
              
            # Guard: refuse backbone-only weights (e.g., .conv.29) for inference/viz
            wpath_lower = str(weights_path).lower()
            if ".conv." in wpath_lower or wpath_lower.endswith(".29"):
                raise RuntimeError(
                    f"Darknet weights appear to be backbone pretraining ('{weights_path}'). "
                    f"Use a trained detector .weights file (e.g., yolov4-tiny.weights or yolov4-tiny-3l.weights)."
                )    
            
            model = mart_yolo_darknet(
                str(cfg_path),
                str(weights_path),
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                detections_per_image=self.max_det,
            )

        elif backend == "yolov4":
            # Note: MART's torchvision-style yolov4 has no pretrained weights by default.
            num_classes = int(getattr(getattr(self.cfg, "model", object()), "num_classes", 80))
            model = mart_yolov4(
                num_classes=num_classes,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                detections_per_image=self.max_det,
            )
            logger.warning("Initialized YOLOv4 backbone without pretrained weights (MART YOLOV4_Weights are stubs).")
        else:
            raise ValueError(f"Unknown MART backend '{backend}'. Use 'darknet' or 'yolov4'.")

        # Move & train mode (we compute losses)
        model.to(device_str)
        model.train()
        self.model = model
        # Sync thresholds in runtime model for consistent inference
        self.model.confidence_threshold = self.confidence_threshold
        self.model.nms_threshold = self.nms_threshold
        self.model.detections_per_image = self.max_det

        # Determine nc (number of classes) and names
        try:
            # Try to introspect from network
            nc = getattr(getattr(self.model, "network", object()), "num_classes", None)
            if nc is None:
                # Fallback: try last detection head attribute (heuristic)
                nc = int(getattr(getattr(self.model, "network", object()), "classes", 80))
        except Exception:
            nc = 80
        self.num_classes = int(nc) if nc is not None else 80
        self._ensure_names(self.num_classes)

        logger.info(f"MART YOLO loaded on {device_str} with nc={self.num_classes} | "
                    f"conf_thres={self.confidence_threshold} nms_thres={self.nms_threshold} max_det={self.max_det}")

    def infer(self, x, target, bboxes, batch_size: int = 1):
        """
        Compute differentiable loss via MART's native YOLO loss (overlap + confidence + classification).
        Inputs:
          x:       Float tensor [N,3,H,W] in [0,1] (no letterbox required; MART handles arbitrary sizes).
          target:  Long[N] class id per image (broadcast if length==1).
          bboxes:  [N,K,4] (or [N,1,4]) xyxy in image pixels.

        Returns:
          loss: scalar tensor (requires_grad=True)
        """
        assert self.model is not None, "Call load_model() first."
        device = next(self.model.parameters()).device

        # Ensure tensor types/devices
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        # x = x.to(device=device, dtype=torch.float32)
        # bboxes = bboxes.to(device=device, dtype=torch.float32)
        # target = target.to(device=device, dtype=torch.long)
        # if x.dim() == 3:
        #     x = x.unsqueeze(0)
        x = x.to(device=device, dtype=torch.float32)
        bboxes = bboxes.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.long)

        # Normalize layout to NCHW (drjit often gives HWC/NHWC)
        if x.dim() == 3 and x.shape[-1] == 3 and x.shape[0] != 3:
            # HWC -> CHW
            x = x.permute(2, 0, 1)
        if x.dim() == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            # NHWC -> NCHW
            x = x.permute(0, 3, 1, 2)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.contiguous()
        
        assert x.dim() == 4 and x.shape[1] == 3, f"Expected NCHW, got {tuple(x.shape)}"
        mart_targets = self._build_targets(x, target, bboxes)

        # Forward through MART. Returns dict with 'detections' and three losses.
        out = self.model(x, mart_targets)
        if not isinstance(out, dict):
            raise RuntimeError("Unexpected MART output. Expected a dict with keys: detections, overlap, confidence, classification.")

        loss = out["overlap"] + out["confidence"] + out["classification"]

        if self.debug:
            logging.getLogger("mart_yolo").info(
                f"MART loss components: overlap={float(out['overlap']):.4f} "
                f"conf={float(out['confidence']):.4f} cls={float(out['classification']):.4f} "
                f"total={float(loss):.4f}"
            )

        return loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=device)

    def _infer_flexible(self, img_3chw: torch.Tensor):
        """
        Accepts [3,H,W], [1,3,H,W], or [1,1,3,H,W]; normalizes to [3,H,W].
        Returns {boxes,scores,labels} tensors or empty tensors on failure.
        """
        logger = logging.getLogger("mart_yolo")
        assert self.model is not None, "load_model() first"
        dev = next(self.model.parameters()).device

        x = img_3chw
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"_infer_flexible expected a torch.Tensor, got {type(x)}")

        # Squeeze optional time/batch leading dims
        if x.dim() == 5 and x.shape[:2] == (1, 1):      # [1,1,3,H,W] -> [3,H,W]
            x = x.squeeze(0).squeeze(0)
        if x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 3:  # [1,3,H,W] -> [3,H,W]
            x = x.squeeze(0)

        # Convert HWC -> CHW if needed
        if x.dim() == 3 and x.shape[-1] == 3 and x.shape[0] != 3:
            x = x.permute(2, 0, 1).contiguous()

        if x.dim() != 3 or x.shape[0] != 3:
            raise RuntimeError(f"_infer_flexible expected [3,H,W], got {tuple(x.shape)}")

        img_3chw = x.to(device=dev, dtype=torch.float32).contiguous()

        # Batch it (unchanged)
        batch = img_3chw.unsqueeze(0)  # [1,3,H,W]

        # ---- Run the **network** directly (bypasses YOLO.forward's loss stacking) ----
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                det_list, _, _ = self.model.network(batch, None)  # (detections_per_layer, losses, hits)

                if not isinstance(det_list, (list, tuple)) or len(det_list) == 0:
                    logger.warning("MART network returned no detection tensors; falling back to empty result.")
                    return {k: torch.empty((0,), device=dev) for k in ("boxes", "scores", "labels")}

                preds = torch.cat(det_list, dim=1)  # [1, anchors, attrs]

                # Optional debug peek
                if self.debug and preds.numel() > 0:
                    conf = preds[..., 4]
                    cls = preds[..., 5:]
                    max_scores = (cls * conf.unsqueeze(-1)).amax(dim=-1)
                    logger.info(f"[DEBUG] pre-filter max score: {float(max_scores.max()):.4f} across {preds.shape[1]} anchors")

                # Use MART's native post-processing (threshold + NMS)
                dets_list = self.model.process_detections(preds)
                dets = dets_list[0] if isinstance(dets_list, (list, tuple)) else dets_list

                # If strict filtering removed everything but we are debugging,
                # return top-K boxes (no NMS) for visualization to help diagnose.
                if (dets is None or dets.get("boxes", torch.empty(0)).numel() == 0) and self.debug and preds.numel() > 0:
                    K = min(20, preds.shape[1])
                    conf = preds[0, :, 4]
                    cls = preds[0, :, 5:]
                    scores_all = cls * conf.unsqueeze(-1)
                    top_scores, top_labels = scores_all.max(dim=-1)
                    topk = torch.topk(top_scores, k=K)
                    boxes = preds[0, topk.indices, :4]
                    scores = topk.values
                    labels = top_labels[topk.indices].to(torch.long)
                    logger.warning("[DEBUG] returning top-K boxes below threshold for viz only")
                    return {"boxes": boxes, "scores": scores, "labels": labels}

                if isinstance(dets, dict):
                    return {
                        "boxes": dets.get("boxes", torch.empty((0, 4), device=dev)).to(device=dev, dtype=torch.float32),
                        "scores": dets.get("scores", torch.empty((0,), device=dev)).to(device=dev, dtype=torch.float32),
                        "labels": dets.get("labels", torch.empty((0,), device=dev)).to(device=dev, dtype=torch.long),
                    }
                else:
                    logger.warning("Unexpected detections structure from process_detections; returning empty result.")
                    return {k: torch.empty((0,), device=dev) for k in ("boxes", "scores", "labels")}
        except Exception as e:
            logger.warning(f"MART infer() failed in _infer_flexible: {e}. Returning empty detections for viz.")
            return {k: torch.empty((0,), device=dev) for k in ("boxes", "scores", "labels")}
        finally:
            if was_training:
                self.model.train()

    # ------------------------- BaseDetector required methods -------------------------
    def preprocess_input(self, image_path: str, *_, **kwargs):
        """
        Load an image for HQ viz. Returns a dict with key 'image' as float32
        tensor [3,H,W] in [0,1]. (Matches what dt2 expects at callsite.)
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(image_path).convert("RGB")
        t = TF.to_tensor(img).contiguous()  # [3,H,W], float in [0,1]

        if t.dim() != 3 or t.shape[0] != 3:
            raise RuntimeError(f"preprocess_input expected RGB -> [3,H,W], got {tuple(t.shape)}")

        return {"image": t, "path": image_path}

    def resolve_label_index(self, label):
        """Map a label (int or str) to an integer class index."""
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, str):
            # try to map via known class names
            if self.class_names is not None:
                try:
                    return int(self.class_names.index(label))
                except ValueError:
                    pass
            # fallback: try to parse as int-like string
            try:
                return int(label)
            except Exception:
                raise ValueError(f"Unknown class label '{label}'. Known: {self.class_names}")
        raise TypeError(f"Unsupported label type: {type(label)}")

    def zero_grad(self):
        """Zero model grads; BaseDetector expects this."""
        if self.model is not None:
            self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def predict_and_save(
        self,
        image: ch.Tensor,
        path: str,
        target: int = None,
        untarget: int = None,
        is_targeted: bool = True,
        threshold: float = 0.25,
        *_,
        **kwargs: Any,
    ):
        """
        Run inference for visualization and save an annotated image.

        Supports panoramic "batch-sensor" renders by splitting the image into
        horizontal tiles and running detection per tile. Pass either:
          - tile_w, tile_h, tiles  (explicit), or
          - resx, resy, sensor_count (aliases used by the caller),
        or it will infer `tiles = W // tile_w` if `tile_w` divides the image width.
        """
        logger = logging.getLogger("mart_yolo")
        assert self.model is not None, "Call load_model() first."

        # Determine working confidence threshold
        conf_thres = float(threshold if threshold is not None else self.confidence_threshold)

        # ---- Accept image as PIL / numpy / torch in CHW or HWC ----
        pil_img = None
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            img_t = TF.to_tensor(pil_img)  # [3,H,W]
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] == 3:
                img_t = torch.from_numpy(image).permute(2, 0, 1).float()
                if img_t.max() > 1.0:
                    img_t = img_t / 255.0
            elif image.ndim == 3 and image.shape[0] == 3:
                img_t = torch.from_numpy(image).float()
                if img_t.max() > 1.0:
                    img_t = img_t / 255.0
            else:
                raise RuntimeError(f"Unsupported numpy image shape: {image.shape}")
        elif isinstance(image, torch.Tensor):
            img_t = image
            if img_t.dim() == 4:
                img_t = img_t[0]
            if img_t.shape[0] != 3 and img_t.shape[-1] == 3:
                img_t = img_t.permute(2, 0, 1)
            img_t = img_t.float()
            if img_t.max() > 1.0:
                img_t = img_t / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Build a base PIL image to draw on
        if pil_img is None:
            np_img = (img_t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pil_img = Image.fromarray(np_img, mode="RGB")
        draw = ImageDraw.Draw(pil_img)
        W, H = pil_img.size

        # ---- Panoramic splitting parameters ----
        # Prefer explicit kwargs, fall back to aliases, then inference.
        tile_w = kwargs.get("tile_w")
        tile_h = kwargs.get("tile_h")
        tiles = kwargs.get("tiles")
        if tile_w is None and "resx" in kwargs:
            tile_w = int(kwargs["resx"])  # alias
        if tile_h is None and "resy" in kwargs:
            tile_h = int(kwargs["resy"])  # alias
        if tiles is None and "sensor_count" in kwargs:
            tiles = int(kwargs["sensor_count"])  # alias

        # Infer tiles if possible
        if tile_w and not tiles:
            tiles = max(1, W // int(tile_w))
        if tile_h is None:
            tile_h = H
        if tiles is None:
            tiles = 1

        # Sanity clamp
        tile_w = int(tile_w) if tile_w else W
        tile_h = int(tile_h)
        tiles = int(tiles)

        # ---- Run detection per tile and collect boxes ----
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        if tiles <= 1 or tile_w >= W:
            dets = self._infer_flexible(img_t)
            boxes = dets.get("boxes", torch.empty(0))
            scores = dets.get("scores", torch.empty(0))
            labels = dets.get("labels", torch.empty(0))
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        else:
            # Split horizontally into `tiles` equal-width crops of width tile_w
            for ti in range(tiles):
                x1 = ti * tile_w
                x2 = min(W, (ti + 1) * tile_w)
                # Slice CHW
                crop = img_t[:, 0:tile_h, x1:x2]
                dets = self._infer_flexible(crop)
                boxes = dets.get("boxes", torch.empty(0))
                scores = dets.get("scores", torch.empty(0))
                labels = dets.get("labels", torch.empty(0))
                if boxes.numel() > 0:
                    # Offset X by tile origin
                    boxes = boxes.clone()
                    boxes[:, [0, 2]] += float(x1)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        # Concatenate results
        if len(all_boxes) == 0:
            boxes = torch.empty((0, 4))
            scores = torch.empty((0,))
            labels = torch.empty((0,), dtype=torch.long)
        else:
            boxes = torch.cat([b for b in all_boxes if b.numel() > 0], dim=0) if any(b.numel() > 0 for b in all_boxes) else torch.empty((0, 4))
            scores = torch.cat([s for s in all_scores if s.numel() > 0], dim=0) if any(s.numel() > 0 for s in all_scores) else torch.empty((0,))
            labels = torch.cat([l for l in all_labels if l.numel() > 0], dim=0) if any(l.numel() > 0 for l in all_labels) else torch.empty((0,), dtype=torch.long)

        # ---- Draw boxes ----
        num_drawn = 0
        if boxes is not None and boxes.numel() > 0:
            try:
                mask = scores >= conf_thres
            except Exception:
                mask = torch.ones_like(scores, dtype=torch.bool)
            boxes_v = boxes[mask].detach().cpu().tolist()
            scores_v = scores[mask].detach().cpu().tolist()
            labels_v = labels[mask].detach().cpu().tolist()

            for (x1, y1, x2, y2), sc, lb in zip(boxes_v, scores_v, labels_v):
                x1 = max(0, min(W - 1, int(round(x1))))
                y1 = max(0, min(H - 1, int(round(y1))))
                x2 = max(0, min(W - 1, int(round(x2))))
                y2 = max(0, min(H - 1, int(round(y2))))
                draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
                name = (
                    self.class_names[lb]
                    if (self.class_names is not None and 0 <= lb < len(self.class_names))
                    else str(lb)
                )
                text = f"{name} {sc:.2f}"
                tw, th = draw.textlength(text), 12
                draw.rectangle([x1, max(0, y1 - th - 2), x1 + int(tw) + 4, y1], fill=(255, 255, 255))
                draw.text((x1 + 2, y1 - th - 1), text, fill=(0, 0, 0))
                num_drawn += 1

        if num_drawn == 0:
            msg = "No detections"
            tw, th = draw.textlength(msg), 12
            draw.rectangle([4, 4, 4 + int(tw) + 6, 4 + th + 4], fill=(255, 255, 255))
            draw.text((7, 6), msg, fill=(0, 0, 0))

        # Ensure destination directory exists and save
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil_img.save(path)

        return {
            "num_detections": int(num_drawn),
            "image_size": (H, W),
            "saved_to": path,
        }
