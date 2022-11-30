import os
import PIL
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision as tv
from torchvision.models import detection
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.detection import MeanAveragePrecision

from typing import Optional, Callable


def visualize(imgs, results, threshold=None, colors=(0, 255, 0)):
    previews = []
    for img, res in zip(imgs, results):
        # res = filter_by_score(res, 0.15)
        if type(res) != dict:
            res = res[0]
        # if threshold is not None:
        #   res = filter_by_score(res, threshold)

        if 'scores' in res:
            _colors = [(colors[0]*l, colors[1]*l, colors[2]*l)
                      for l in res['scores']]
            labels = [f"{int(i)}:{np.round(float(s), 2)}" for i,
                      s in zip(res['labels'], res['scores'])]
        else:
            _colors = colors
            # colors = [(colors[0], colors[1], colors[2]) for _ in res['labels']]
            # print(colors)
            labels = [str(int(i)) for i in res['labels']]

        preview = tv.utils.draw_bounding_boxes(
            (255*img).to(torch.uint8),
            res['boxes'],
            labels=labels,
            colors=_colors
        )

        previews.append(preview)

    return tv.utils.make_grid(previews, nrow=int(np.sqrt(len(previews))))


def collate_fn(batch):
    # https://github.com/pytorch/vision/blob/4a310f26049371959617921d0eb9b001f4d262c6/references/detection/utils.py#L203
    imgs, targets = tuple(zip(*batch))
    imgs = torch.stack(imgs)
    return imgs, targets


class AmougDataset(tv.datasets.VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, gen_masks=False) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.gen_masks = gen_masks

    def __len__(self):
        # count number of file which start with img
        return sum(file_name.startswith("img") for file_name in os.listdir(self.root))

    def __getitem__(self, index: int):
        # Load the image and mask
        img_path = os.path.join(self.root, f"img{index}.png")
        mask_path = os.path.join(self.root, f"label{index}.png")

        img = PIL.Image.open(img_path).convert("RGB")
        mask = PIL.Image.open(mask_path)
        mask = pil_to_tensor(mask)

        # Split color encoded image into binary masks
        obj_ids = torch.unique(mask)[1:]
        num_objs = len(obj_ids)
        masks = mask == obj_ids[:, None, None]

        # Get bounding box coordinate from mask
        boxes = tv.ops.masks_to_boxes(masks)

        # Construct a target
        target = {
            'boxes': boxes,
            # there is only amougus
            'labels': torch.ones((num_objs,), dtype=torch.int64),
            'image_id': torch.tensor([index]),
        }
        if self.gen_masks:
            target['masks'] = masks

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class AmougRCNNModel(pl.LightningModule):
    def __init__(self, lr=0.005, momentum=0.9, weight_decay=0.0005):
        super().__init__()

        # Load the base model
        model = detection.fasterrcnn_resnet50_fpn_v2(
            weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            progress=True
        )
        # Freeze model's parameters
        for params in model.parameters():
            params.requires_grad = False

        # Replace model's head box predictor with our owns
        # Get size of box predictor input
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        new_box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, 2)  # Create new box predictor
        # Replace model's head box predictor with our owns
        model.roi_heads.box_predictor = new_box_predictor

        self.model = model

        # Metric for testing step
        self.test_map = MeanAveragePrecision()

        # Save hyperparameter in checkpoints
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        losses = self.model(imgs, targets)
        loss = sum(losses.values())
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        self.model.train()
        losses = self.model(imgs, targets)
        loss = sum(losses.values())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.model(imgs, targets)

        value = self.test_map(preds, targets)
        self.log('test_map', value)

    def test_epoch_end(self, outputs):
        self.test_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return optimizer
