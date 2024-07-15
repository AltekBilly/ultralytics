# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import Altek_LandmarkModel, generate_fuse_list
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results

import torch.quantization as quant
import torch

class Altek_LandmarkTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a altek_landmark model.

    Example:
        ```python
        from ultralytics.models.yolo.altek_landmark import Altek_LandmarkTrainer

        args = dict(model='yolov8n-altek_landmark.pt', data='coco8-altek_landmark.yaml', epochs=3)
        trainer = Altek_LandmarkTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a Altek_LandmarkTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "altek_landmark"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Altek_Landmark bug. Recommend 'device=cpu' for Altek_Landmark models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

        self.qat = overrides['qat']
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get altek_landmark estimation model with specified configuration and weights."""
        model = Altek_LandmarkModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)

        if self.qat:
            default_qconfig = quant.get_default_qat_qconfig('qnnpack')
            custom_qconfig = quant.QConfig(
                activation=CustomObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                weight=default_qconfig.weight)    
            _quant   = quant.QuantStub()
            _dequant = quant.DeQuantStub()
            _quant.qconfig   = custom_qconfig
            _dequant.qconfig = custom_qconfig
            _quant = quant.prepare_qat(_quant)
            # _dequant = quant.prepare_qat(_dequant)
            
            list = [_quant]
            model.model[-1].qat = True
            for idx, m in enumerate(model.model):
                fuse_list = generate_fuse_list(m)
                m = quant.fuse_modules(m.eval(), fuse_list)
                m.qconfig = default_qconfig #custom_qconfig

                # if idx == len(model.model) - 1:
                #     m.DeQuantStub = _dequant
                
                m = quant.prepare_qat(m.train(), inplace=True)
                list.append(m)  
            
            model.model = torch.nn.Sequential(*list)
            
        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of Altek_LandmarkModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the Altek_LandmarkValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"#, "nme_loss"
        return yolo.altek_landmark.Altek_LandmarkValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
            
    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

class CustomObserver(quant.MinMaxObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_scale = torch.tensor([1.0 / 255])
        self.custom_zero_point = torch.tensor([0])

    def calculate_qparams(self):
        return self.custom_scale, self.custom_zero_point