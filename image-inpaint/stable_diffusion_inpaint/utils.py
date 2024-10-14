import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple


def crop_for_fill_pred(
    image: np.array,
    mask: np.array,
    crop_size: int = 512):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    # If image shorter side is less than 512
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)
    
        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))
    
    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    height, width = image.shape[:2]

    # If 512 cannot cover the entire mask
    if w > crop_size or h > crop_size:
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)
    
    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

    # Crop the image
    cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    cropped_mask = mask[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return cropped_image, cropped_mask


def crop_for_fill_post(
        image: np.array,
        mask: np.array,
        filled_image: np.array, 
        crop_size: int = 512,):
    image_copy = image.copy()
    mask_copy = mask.copy()
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    height_ori, width_ori = height, width
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than 512, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Update the height and width of the resized image
    height, width = image.shape[:2]

    # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        flag_padding = True
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
            padding_side = 'h'
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')
            padding_side = 'w'

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)
    else:
        flag_padding = False

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

    # Fill the image
    image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = filled_image
    if flag_padding:
        image = cv2.resize(image, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
        if padding_side == 'h':
            image = image[padding // 2:padding // 2 + height_ori, :]
        else:
            image = image[:, padding // 2:padding // 2 + width_ori]

    image = cv2.resize(image, (width_ori, height_ori))

    image_copy[mask_copy==255] = image[mask_copy==255]
    return image_copy


def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes an image and its corresponding mask to have the longer side equal to `target_size` and pads them to make them
    both have the same size. The resulting image and mask have dimensions (target_size, target_size).

    Args:
        image: A numpy array representing the image to resize and pad.
        mask: A numpy array representing the mask to resize and pad.
        target_size: An integer specifying the desired size of the longer side after resizing.

    Returns:
        A tuple containing two numpy arrays - the resized and padded image and the resized and padded mask.
    """
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    mask_padded = np.pad(mask_resized, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    return image_padded, mask_padded, (top_pad, bottom_pad, left_pad, right_pad)


def recover_size(image_padded: np.ndarray, mask_padded: np.ndarray, orig_size: Tuple[int, int], 
                 padding_factors: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes a padded and resized image and mask to the original size.

    Args:
        image_padded: A numpy array representing the padded and resized image.
        mask_padded: A numpy array representing the padded and resized mask.
        orig_size: A tuple containing two integers - the original height and width of the image before resizing and padding.

    Returns:
        A tuple containing two numpy arrays - the recovered image and the recovered mask with dimensions `orig_size`.
    """
    h,w,c = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors
    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    mask = mask_padded[top_pad:h-bottom_pad, left_pad:w-right_pad]
    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    return image_resized, mask_resized