# Copyright (c) Aleksandr Fedorov

# Intended to use in the interactive mode

import torch

FILE = "/home/fedorov/detection/maskrcnn-benchmark/traffic/models/COCOPretrained/R-50-GN_no_head.pth"
state_dict = torch.load(FILE)
print(state_dict.keys())


# Pick proper keys based on the loaded state dictionary

# for k in ['roi_heads.box.predictor.cls_score.weight', 'roi_heads.box.predictor.cls_score.bias',
#           'roi_heads.box.predictor.bbox_pred.weight', 'roi_heads.box.predictor.bbox_pred.bias',
#           'roi_heads.mask.predictor.mask_fcn_logits.weight', 'roi_heads.mask.predictor.mask_fcn_logits.bias']:
#     k = "module."+k
#     del state_dict['model'][k]


# for k in ['roi_heads.box.predictor.cls_score.weight', 'roi_heads.box.predictor.cls_score.bias',
#           'roi_heads.box.predictor.bbox_pred.weight', 'roi_heads.box.predictor.bbox_pred.bias',
#           'roi_heads.mask.predictor.mask_fcn_logits.weight', 'roi_heads.mask.predictor.mask_fcn_logits.bias']:
#     k = "model."+k
#     del state_dict['model'][k]


# for k in ["cls_score.bias", "cls_score.weight", "mask_fcn_logits.bias", "mask_fcn_logits.weight",
#           "bbox_pred.bias", "bbox_pred.weight"]:
#     del state_dict['model'][k]


del state_dict['optimizer']
del state_dict['scheduler']
del state_dict['iteration']
# file = file.replace('.pth', '_no_head.pth')
FILE = FILE.replace('.pkl', '_no_head.pkl')
torch.save(state_dict, FILE)