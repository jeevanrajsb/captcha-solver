import cv2
import numpy as np

def check_letter_tarakom(pic):
    """
    Check if a letter region meets specific density criteria.

    Args:
        pic (numpy.ndarray): Image region to check.

    Returns:
        bool: True if letter meets density criteria, False otherwise.
    """
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
    return total - howmanyblack <= 22

def preprocess_letter(letter_crop):
    """
    Preprocess a letter image for model prediction.

    Args:
        letter_crop (numpy.ndarray): Cropped letter region.

    Returns:
        numpy.ndarray: Preprocessed letter image.
    """
    resized_letter = cv2.resize(letter_crop, (32, 52))
    resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
    return binarized
