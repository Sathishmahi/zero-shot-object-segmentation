import torch
import os
import supervision as sv
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"


sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_predictor = SamPredictor(sam)



def predictor(frame,bb,draw_bb=True):
  mask_predictor.set_image(frame)
  masks, scores, logits = mask_predictor.predict(
      box=bb,
      multimask_output=True
  )
  mask_annotator = sv.MaskAnnotator(color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)
  detections = sv.Detections(
      xyxy=sv.mask_to_xyxy(masks=masks),
      mask=masks
  )
  detections = detections[detections.area == np.max(detections.area)]
  segmented_image = mask_annotator.annotate(scene=frame.copy(), detections=detections)
  
  return segmented_image