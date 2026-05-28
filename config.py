import numpy as np

# ===== VOC Classes =====

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train',
    'tv/monitor'
]

# ===== RGB Colour Palette =====

PALETTE = np.array([
    [0,0,0],
    [135,206,235],
    [255,165,0],
    [255,215,0],
    [0,191,255],
    [148,0,211],
    [255,20,147],
    [220,20,60],
    [255,140,0],
    [139,69,19],
    [255,255,0],
    [210,105,30],
    [186,85,211],
    [255,105,180],
    [0,255,127],
    [255,69,0],
    [34,139,34],
    [240,230,140],
    [0,206,209],
    [0,0,255],
    [127,255,212]
], dtype=np.uint8)

# ===== Decoder =====

def decode_segmap(seg_mask):
    rgb = PALETTE[seg_mask % len(PALETTE)]
    return rgb