import torch as ch
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
log = logging.getLogger(__name__)

class BaseDetector(ABC):
    """
    A base class for object detectors that provides a common interface:
        - load_model(): Loads and initializes the detection model
        - infer(image: Any) -> List[Dict]: Takes a single image or a batch of images
                                           and returns a list of detection results,
                                           each result being:
                                              {
                                                "bbox": [x1, y1, x2, y2],
                                                "score": float,
                                                "label": int
                                              }
    Subclasses must implement these methods.
    """

    def __init__(self):
        # By default, detectors expect [0,1] floats for predict_and_save inputs
        self.expects_unit_input = True
    @abstractmethod
    def load_model(self):
        """
        Load the detection model, including weights and configurations.
        """
        pass

    @abstractmethod
    def infer(self, x: ch.Tensor, target: ch.Tensor, bboxes: ch.Tensor, batch_size: int = 1) -> ch.Tensor:
        """
        Run model inference for loss computation.
 
        Args:
            x: Input image tensor of shape (C, H, W) or (N, C, H, W)
            target: Target class labels
            bboxes: Ground truth bounding boxes
            batch_size: Optional batch size
 
        Returns:
            torch.Tensor: Loss value for the input and target.
        """
        pass

    @abstractmethod
    def predict_and_save(self, image: ch.Tensor, path: str, target: int = None, untarget: int = None, is_targeted: bool = True, threshold: float = 0.7, format: str = "RGB", gt_bbox: List[int] = None, result_dict: bool = False) -> Any:
        """
        Run model prediction on the given image and save the visualization to disk.

        Args:
            image: The input image as a tensor.
            path: Output path to save the prediction visualization.
            target: Class ID to confirm presence of (for targeted attack eval).
            untarget: Class ID to confirm absence of (for untargeted attack eval).
            is_targeted: Whether evaluation is for targeted attack.
            threshold: Confidence threshold for visualized predictions.
            format: Image color format for visualization (RGB or BGR).
            gt_bbox: Optional ground truth bounding box [x1, y1, x2, y2] for visualization or evaluation purposes.

        Returns:
            bool: Whether the prediction meets the target/untarget criteria.
            If `result_dict` is True, also returns:
                Dict[str, Any]: {
                    "closest_class": int,
                    "closest_confidence": float
                }
        """
        pass

    @abstractmethod
    def preprocess_input(self, image_path: str) -> Dict[str, Any]:
        """
        Convert an image path into a model-specific input dictionary.

        Args:
            image_path: Path to the image file.

        Returns:
            A dictionary with keys expected by the model, such as:
                - image: Tensor
                - height: int
                - width: int
                - instances: (optional) GT boxes and labels
        """
        pass

    @abstractmethod
    def zero_grad(self):
        """
        Zero out gradients of the detector model.
        """
        pass

    @abstractmethod
    def resolve_label_index(self, label_name: str) -> int:
        """
        Converts a human-readable class name (e.g., 'person') into a model-specific label index.
        """
        pass