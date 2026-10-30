
import cv2
import numpy as np
from PIL import Image


def preprocess_food_image(image):
    """
    Prepare a food image for computer vision analysis.
    Returns original, resized, denoised and enhanced images.
    """
    image = image.convert("RGB")
    original = np.array(image)

    # Resize to a consistent size
    resized = cv2.resize(original, (224, 224))

    # Reduce minor image noise
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)

    # Improve local contrast
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )
    enhanced = cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2RGB
    )

    return {
        "Original": original,
        "Resized (224×224)": resized,
        "Denoised": denoised,
        "Contrast Enhanced": enhanced,
    }
