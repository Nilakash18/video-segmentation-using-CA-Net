
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import time
from tqdm import tqdm

OUTPUT_FPS = 25

OUTPUT_SEG_VIDEO = "outputs/seg.mp4"
OUTPUT_BLEND_VIDEO = "outputs/blend.mp4"
OUTPUT_SIDE_VIDEO = "outputs/side.mp4"

from config import (
    VOC_CLASSES,
    PALETTE,
    decode_segmap
)

from model import (
    seg_model,
    preprocess,
    DEVICE
)

# ===== Inference Size =====

INFER_SIZE = 520

# ===== Segment Frame =====

def segment_frame(frame_bgr, alpha=0.55):

    h, w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    pil_img = Image.fromarray(frame_rgb)

    inp = preprocess(
        pil_img
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        out = seg_model(inp)["out"]

        out = F.interpolate(
            out,
            size=(h,w),
            mode='bilinear',
            align_corners=False
        )

        pred = (
            out.argmax(dim=1)
            .squeeze()
            .cpu()
            .numpy()
        )

    pred_resized = cv2.resize(
        pred.astype(np.uint8),
        (w,h),
        interpolation=cv2.INTER_NEAREST
    )

    seg_rgb = decode_segmap(
        pred_resized
    )

    seg_bgr = cv2.cvtColor(
        seg_rgb,
        cv2.COLOR_RGB2BGR
    )

    blend_bgr = cv2.addWeighted(
        frame_bgr,
        1-alpha,
        seg_bgr,
        alpha,
        0
    )

    detected = (
        set(np.unique(pred_resized).tolist())
        - {0}
    )

    return seg_rgb, blend_bgr, detected

# ===== Legend Bar =====

def make_legend_bar(
        class_ids,
        bar_w,
        bar_h=40
):

    bar = np.zeros(
        (bar_h, bar_w, 3),
        dtype=np.uint8
    )

    classes = sorted(class_ids)

    if not classes:
        return bar

    sw = bar_w // max(
        len(classes),
        1
    )

    for i, cid in enumerate(classes):

        x0 = i * sw
        x1 = min(
            (i+1)*sw,
            bar_w
        )

        color = PALETTE[
            cid % len(PALETTE)
        ].tolist()

        bar[:,x0:x1] = color

        label = (
            VOC_CLASSES[cid]
            if cid < len(VOC_CLASSES)
            else str(cid)
        )

        cv2.putText(
            bar,
            label,
            (x0+3, bar_h-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0,0,0),
            1,
            cv2.LINE_AA
        )

    return bar

# CELL 10: Process Full Video (CA-DeepLab)
# ============================================================

def process_video(video_path,
                  alpha=0.5,
                  max_frames=None,
                  frame_skip=1,
                  show_legend=True,
                  show_frame_num=True,
                  resize_output=None):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if resize_output:
        out_w, out_h = resize_output
    else:
        out_w, out_h = W, H

    side_w = out_w * 3

    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    vw_seg = cv2.VideoWriter(
        OUTPUT_SEG_VIDEO,
        fourcc,
        OUTPUT_FPS,
        (out_w,out_h)
    )

    vw_blend = cv2.VideoWriter(
        OUTPUT_BLEND_VIDEO,
        fourcc,
        OUTPUT_FPS,
        (out_w,out_h)
    )

    vw_side = cv2.VideoWriter(
        OUTPUT_SIDE_VIDEO,
        fourcc,
        OUTPUT_FPS,
        (side_w,out_h)
    )

    frame_idx = 0
    write_idx = 0

    all_classes = set()

    start_time = time.time()

    n_process = (
        min(max_frames,n_tot)
        if max_frames else n_tot
    )

    pbar = tqdm(
        total=max(1,n_process//frame_skip),
        desc='Segmenting',
        unit='frame'
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # -------- segmentation --------
        seg_rgb, blend_bgr, detected = segment_frame(
            frame,
            alpha=alpha
        )

        all_classes |= detected

        seg_bgr = cv2.cvtColor(
            seg_rgb,
            cv2.COLOR_RGB2BGR
        )

        # -------- resize --------
        if resize_output:

            frame = cv2.resize(
                frame,
                (out_w,out_h)
            )

            seg_bgr = cv2.resize(
                seg_bgr,
                (out_w,out_h),
                interpolation=cv2.INTER_NEAREST
            )

            blend_bgr = cv2.resize(
                blend_bgr,
                (out_w,out_h)
            )

        # -------- legend --------
        if show_legend and detected:

            legend = make_legend_bar(
                detected,
                out_w,
                bar_h=36
            )

            legend_bgr = cv2.cvtColor(
                legend,
                cv2.COLOR_RGB2BGR
            )

            blend_bgr[-36:,:] = cv2.addWeighted(
                blend_bgr[-36:,:],
                0.30,
                legend_bgr,
                0.70,
                0
            )

        # -------- frame counter --------
        if show_frame_num:

            txt = f"Frame {write_idx+1}"

            for img in [frame,seg_bgr,blend_bgr]:

                cv2.putText(
                    img,
                    txt,
                    (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )

        # -------- labels --------
        labels = [
            (frame,'ORIGINAL'),
            (seg_bgr,'SEGMENTATION'),
            (blend_bgr,'CA-DEEPLAB')
        ]

        for img,label in labels:

            cv2.putText(
                img,
                label,
                (10,out_h-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255,255,255),
                2
            )

        # -------- save --------
        vw_seg.write(seg_bgr)

        vw_blend.write(blend_bgr)

        side = np.concatenate(
            [frame,seg_bgr,blend_bgr],
            axis=1
        )

        vw_side.write(side)

        frame_idx += 1
        write_idx += 1

        pbar.update(1)

    pbar.close()

    cap.release()

    vw_seg.release()
    vw_blend.release()
    vw_side.release()

    elapsed = time.time() - start_time

    print(f"\n✅ Done!")

    print(
        f"Processed {write_idx} frames "
        f"in {elapsed:.1f}s"
    )

    print(
        f"Average FPS: "
        f"{write_idx/max(elapsed,1):.2f}"
    )

    print("\nDetected Classes:")

    print(
        ", ".join(
            [
                VOC_CLASSES[c]
                for c in sorted(all_classes)
                if c < len(VOC_CLASSES)
            ]
        )
    )

    return all_classes