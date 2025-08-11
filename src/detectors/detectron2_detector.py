from typing import List, Dict, Any
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from .base_detector import BaseDetector
import os, PIL
import torch as ch
from torchvision.io import read_image
import logging
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
from detectron2.data.detection_utils import *
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from torchvision.ops import box_iou


class Detectron2Detector(BaseDetector):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg            # The full Hydra/OmegaConf config
        self.dt2_cfg = None       # Will hold the Detectron2 config after load_model
        self.model = None

    def load_model(self):
        """
        Initializes and configures a Detectron2 model for object detection using self.cfg.
        """
        model_config = "pretrained-models/faster_rcnn_R_50_FPN_3x/config.yaml"
        weights_file = "pretrained-models/faster_rcnn_R_50_FPN_3x/model_final.pth"
        score_thresh = 0.5
        DEVICE = f"cuda:{self.cfg.device}" 

        logging.basicConfig(level=logging.INFO)
        detectron_cfg = get_cfg()
        detectron_cfg.merge_from_file(model_config)
        detectron_cfg.MODEL.WEIGHTS = weights_file
        detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        detectron_cfg.MODEL.DEVICE = DEVICE

        self.dt2_cfg = detectron_cfg
        self.model = build_model(self.dt2_cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.dt2_cfg.MODEL.WEIGHTS)

        self.model.train = True
        self.model.training = True
        self.model.proposal_generator.training = True
        self.model.roi_heads.training = True

    def model_train_mode(self):
        self.model.train = True
        self.model.training = True
        self.model.proposal_generator.training = True
        self.model.roi_heads.training = True
    
    def model_eval_mode(self):
        self.model.train = False
        self.model.training = False
        self.model.proposal_generator.training = False
        self.model.roi_heads.training = False
    
    def infer(self, x, target, bboxes, batch_size=1):
        """
        Computes and returns the classification loss for the input image tensor and target.
        """
        self.model_train_mode()     
        if x.dim() == 3:
            x = x.unsqueeze(0).requires_grad_()
        x.retain_grad()

        losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]
        target_loss_idx = [0]
        x = ch.clip(x * 255 + 0.5, 0, 255).requires_grad_()
        x.retain_grad()
        height = x.shape[2]
        width = x.shape[3]
        # Normalize GT boxes to shape [N, 1, 4] on the same device/dtype as x
        if isinstance(bboxes, ch.Tensor):
            gt = bboxes.detach().to(device=x.device, dtype=ch.float32)
        else:
            gt = ch.as_tensor(bboxes, dtype=ch.float32, device=x.device)

        if gt.dim() == 1:
            # [4] -> [1, 1, 4]
            gt_boxes = gt.view(1, 1, 4)
        elif gt.dim() == 2:
            # [N,4] -> [N,1,4]
            if gt.shape[1] != 4:
                raise RuntimeError(f"Expected bboxes second dimension to be 4, got {gt.shape}")
            gt_boxes = gt.unsqueeze(1)
        elif gt.dim() == 3:
            # assume already [N,1,4] or [N,K,4]; we will index [i]
            if gt.shape[-1] != 4:
                raise RuntimeError(f"Expected last bbox dimension to be 4, got {gt.shape}")
            gt_boxes = gt
        else:
            raise RuntimeError(f"Unexpected bboxes ndim: {gt.dim()}")

        inputs = []
        for i in range(x.shape[0]):
            instances = Instances(image_size=(height, width))
            instances.gt_classes = target.long()
            instances.gt_boxes = Boxes(gt_boxes[i])
            input_dict = {
                'image': x[i],
                'filename': '',
                'height': height,
                'width': width,
                'instances': instances
            }
            inputs.append(input_dict)

        with EventStorage(0):
            losses = self.model(inputs)
            loss = sum([losses[losses_name[tgt_idx]] for tgt_idx in target_loss_idx]).requires_grad_()
        del x
        return loss

    def predict_and_save(self, image: ch.Tensor, path: str, target: int = None, untarget: int = None, is_targeted: bool = True, threshold: float = 0.7, format: str = "RGB", gt_bbox: List[int] = None, result_dict: bool = False, image_id: int = None) -> Any:
        """
        Run model prediction on the given image and save the visualization to disk.
        """
        self.model_eval_mode()
        with ch.no_grad():
            height = image.shape[1]
            width = image.shape[2]
            input = {
                'image': (image * 255).to(dtype=ch.uint8),
                'height': height,
                'width': width,
                'instances': Instances((height, width))
            }
            input['instances'].gt_classes = ch.Tensor([2])
            input['instances'].gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))

            outputs = self.model([input])
            instances = outputs[0]['instances']
            mask = instances.scores > threshold
            instances = instances[mask]

            # Convert tensor image to numpy for visualization
            pbi = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if format == "BGR":
                pbi = pbi[:, :, ::-1]

            v = Visualizer(pbi, MetadataCatalog.get(self.dt2_cfg.DATASETS.TRAIN[0]), scale=1.0)
            things = np.array(MetadataCatalog.get(self.dt2_cfg.DATASETS.TRAIN[0]).thing_classes)
            predicted_classes = things[instances.pred_classes.cpu().numpy().tolist()]
            # log.info(f'Predicted Class: {predicted_classes}')
            out = v.draw_instance_predictions(instances.to("cpu"))

            closest_confidence = None
            best_class = None
            best_iou = None
            formatted_gt_bbox = None
            best_idx = None
            if gt_bbox is not None:
                # FIXME: makeoption to draw gt bbox
                # v.draw_box(gt_bbox, edge_color='green')

                gt_box_tensor = ch.tensor([gt_bbox], dtype=ch.float32)
                # compute IoU between each predicted box and the single GT box
                pred_boxes = instances.pred_boxes
                if len(pred_boxes) > 0:
                    pred_t = instances.pred_boxes.tensor  # [N,4]
                    gt_t = gt_box_tensor.to(pred_t.device)  # [1,4]
                    ious_mat = box_iou(pred_t, gt_t)        # [N,1]
                    ious = ious_mat[:, 0]                  # [N]
                    best_idx = ious.argmax().item()
                    best_iou = float(ious[best_idx].item())
                    best_class = int(instances.pred_classes[best_idx].item()) if best_iou > 0.5 else None
                    closest_confidence = float(instances.scores[best_idx].item()) if best_iou > 0.5 else None
                    target_pred_exists = (best_iou > 0.5 and best_class == target)
                else:
                    target_pred_exists = False
            else:
                target_pred_exists = False

            if gt_bbox is not None:
                gt_box_tensor = ch.tensor([gt_bbox], dtype=ch.float32)
                x1, y1, x2, y2 = [float(coord) for coord in gt_box_tensor[0]]
                w = x2 - x1
                h = y2 - y1
                formatted_gt_bbox = [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)]
                # compute IoU again for untarget check
                pred_t = instances.pred_boxes.tensor  # [N,4]
                gt_t   = gt_box_tensor.to(pred_t.device)  # [1,4]
                if pred_t.numel() > 0:
                    ious_mat = box_iou(pred_t, gt_t)   # [N,1]
                    ious = ious_mat[:, 0]
                    best_idx = ious.argmax().item()
                    best_iou = float(ious[best_idx].item())
                    best_class = int(instances.pred_classes[best_idx].item()) if best_iou > 0.5 else None
                    untarget_pred_not_exists = not (best_iou > 0.5 and best_class == untarget)
                else:
                    untarget_pred_not_exists = True
            else:
                untarget_pred_not_exists = all(cls != untarget for cls in instances.pred_classes.cpu().numpy())

            class_names = MetadataCatalog.get(self.dt2_cfg.DATASETS.TRAIN[0]).thing_classes
            best_class_name = class_names[best_class] if best_class is not None else None

        self.model_train_mode()
        pred = out.get_image()
        PIL.Image.fromarray(pred).save(path)

        if result_dict:
            # assemble structured detections list (COCO format) from Detectron2 outputs
            detections = []
            pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
            pred_scores = instances.scores.cpu().numpy()
            pred_classes = instances.pred_classes.cpu().numpy()
            for idx in range(len(pred_boxes)):
                x1, y1, x2, y2 = pred_boxes[idx]
                w = x2 - x1
                h = y2 - y1
                bbox = [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)]
                detection = {
                    "image_id": getattr(self, 'image_id', -1),
                    "category_id": int(pred_classes[idx]),
                    "bbox": bbox,
                    "score": float(pred_scores[idx])
                }
                detections.append(detection)

            meets_criteria = (
                (is_targeted and target_pred_exists and (untarget is None or untarget_pred_not_exists)) or
                (not is_targeted and untarget_pred_not_exists)
            )
            return_result = {
                "detections": detections,
                "closest_class": best_class if gt_bbox is not None else None,
                "closest_class_name": best_class_name if gt_bbox is not None and len(pred_boxes) > 0 else None,
                "closest_category_id": best_class if best_class is not None else None,
                "closest_confidence": closest_confidence if gt_bbox is not None else None,
                "closest_bbox": [float(x) for x in detections[best_idx]["bbox"]] if gt_bbox is not None and best_idx is not None else None,
                "gt_bbox": [float(x) for x in formatted_gt_bbox] if formatted_gt_bbox is not None else None,
                "best_iou": float(best_iou) if best_iou is not None else None,
                "target_pred_exists": target_pred_exists,
                "untarget_pred_not_exists": untarget_pred_not_exists,
            }
            return meets_criteria, return_result

        if is_targeted:
            if target_pred_exists and untarget is None:
                return True
            elif target_pred_exists and untarget is not None and untarget_pred_not_exists:
                return True
        elif not is_targeted and untarget_pred_not_exists:
            return True
        return False

    def preprocess_input(self, image_path: str) -> Dict[str, Any]:
        """
        Construct a Detectron2-friendly input for an image.
        """
        input = {}
        filename = image_path
        adv_image = read_image(image_path, format="RGB")
        adv_image_tensor = ch.as_tensor(np.ascontiguousarray(adv_image.transpose(2, 0, 1)))

        height = adv_image_tensor.shape[1]
        width = adv_image_tensor.shape[2]
        instances = Instances(image_size=(height, width))
        instances.gt_classes = ch.Tensor([2])  # Default placeholder; not always used
        instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))

        input['image'] = adv_image_tensor
        input['filename'] = filename
        input['height'] = height
        input['width'] = width
        input['instances'] = instances
        return input
    
    def zero_grad(self):
        """
        Zero out the gradients of the Detectron2 model.
        """
        if self.model is not None:
            self.model.zero_grad()
            # for param in self.model.parameters():
            #     if param.grad is not None:
            #         param.grad.zero_()

    def resolve_label_index(self, label_name: str) -> int:
        """
        Converts a human-readable class name into a Detectron2 label index based on COCO metadata.
        """
        def normalize(name):
            return name.replace('_', ' ').lower()

        label_name = normalize(label_name)
        class_names = MetadataCatalog.get(self.dt2_cfg.DATASETS.TRAIN[0]).thing_classes
        label_lookup = {normalize(name): idx for idx, name in enumerate(class_names)}

        if label_name not in label_lookup:
            raise ValueError(f"Label '{label_name}' not found in Detectron2 class list.")

        return label_lookup[label_name]
