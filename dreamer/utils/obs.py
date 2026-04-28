"""
Observation preprocessing — numpy/cv2 only (no torch), safe for Pi5.

preprocess_frame() is the single source of truth for the camera→observation
pipeline used by sim (env.py), real car (drive_physical_tflite.py), and agent.
"""

import cv2
import numpy as np

CROP_TOP  = 40      # rows to drop from top of frame (sky)
IMG_SIZE  = 64      # output spatial size (square)


def preprocess_frame(frame: np.ndarray, channels: int = 1) -> np.ndarray:
    """
    [H, W, 3] uint8 → [C, 64, 64] float32 in [-0.5, 0.5].

    Crops the top CROP_TOP rows (sky), resizes to IMG_SIZE×IMG_SIZE.
    channels=1 → grayscale  (1, 64, 64)
    channels=3 → RGB        (3, 64, 64)
    """
    frame = frame[CROP_TOP:, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    if channels == 1:
        frame = np.dot(frame, [0.299, 0.587, 0.114]).astype(np.float32)
        return (frame / 255.0 - 0.5)[np.newaxis]          # (1, 64, 64)
    return (frame.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)  # (3, 64, 64)
