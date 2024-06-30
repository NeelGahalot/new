# Checking
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.transforms as transforms

def crf_with_prob(img, mask, prob):
    """
    Applies DenseCRF to refine the segmentation mask.
    
    Args:
    - img (numpy array): The original image.
    - mask (numpy array): The initial segmentation mask.
    - prob (numpy array): The probability array indicating likelihood of pixels being in the foreground.
    
    Returns:
    - refined_mask (numpy array): The refined segmentation mask.
    """
    # Convert image to numpy array if it's a tensor
    if isinstance(img, torch.Tensor):
        img = img.numpy().transpose(1, 2, 0)

    # Get height and width of image
    h, w = img.shape[:2]

    # Ensure mask is a 2D array
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[0] == 3:  # In case the mask is RGB
        mask = mask[0]

    # Ensure mask has the same height and width as the image
    assert mask.shape == (h, w), f"Mask shape {mask.shape} does not match image shape {(h, w)}"

    # Initialize DenseCRF
    d = dcrf.DenseCRF2D(w, h, 2)

    # Create unary potentials
# Create unary potentials
    mask = mask.astype(np.float32) / 255.0  # Normalize mask to range [0, 1]
    #unary_bg = 1 - prob  # Background unary potential
    #unary_fg = prob  # Foreground unary potential
    unary = np.stack([prob, 1 - prob], axis=0) 
    unary = unary.reshape((2, -1))
    #unary = np.stack([unary_bg, unary_fg], axis=-1)  # Stack to create unary potentials
    d.setUnaryEnergy(unary)


    # Create pairwise potentials
    pairwise_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(pairwise_gaussian, compat=3)

    pairwise_bilateral = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=img, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Perform inference
    Q = d.inference(5)
    refined_mask = np.argmax(Q, axis=0).reshape((h, w))

    return refined_mask

def apply_dense_crf(img, mask):
    """
    Applies DenseCRF to refine the segmentation mask.
    
    Args:
    - img (numpy array): The original image.
    - mask (numpy array): The initial segmentation mask.
    
    Returns:
    - refined_mask (numpy array): The refined segmentation mask.
    """
    # Convert image to numpy array if it's a tensor
    if isinstance(img, torch.Tensor):
        print('lala')
        img = img.numpy().transpose(1, 2, 0)

    # Get height and width of image
    h, w = img.shape[:2]

    # Ensure mask is a 2D array
    if mask.ndim == 3 and mask.shape[0] == 1:
        print('hi')
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[0] == 3:
        print('hello')
        # In case the mask is RGB
        mask = mask[0]

    # Ensure mask has the same height and width as the image
    assert mask.shape == (h, w), f"Mask shape {mask.shape} does not match image shape {(h, w)}"

    # Initialize DenseCRF
    d = dcrf.DenseCRF2D(w, h, 2)

    # Create unary potentials
    mask = mask.astype(np.float32) / 255.0  # Normalize mask to range [0, 1]
    unary = np.stack([mask, 1 - mask], axis=0)  # Swapped to correctly represent foreground and background
    unary = unary.reshape((2, -1))  # Shape should be (num_classes, height*width)
    d.setUnaryEnergy(unary)

    # Create pairwise potentials
    pairwise_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(pairwise_gaussian, compat=3)

    pairwise_bilateral = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=img, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Perform inference
    Q = d.inference(5)
    refined_mask = np.argmax(Q, axis=0).reshape((h, w))

    return refined_mask

def polygon_vertices(mask):
    '''
    Find contours in the binary image

    mask: binary image to find contours in
    '''
    # Convert the image into 0 and 1 (background and foreground)
    _, binary_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extract segmentations
    segmentations = []
    for contour in contours:
        # Approximate contour to polygon
        if len(contour) < 3:
            continue

        contour = cv2.approxPolyDP(contour, 1, True)
        segmentation = contour.flatten().tolist()
        segmentations.append(segmentation)

    return segmentations

def calculate_polygon_area(vertices):
    # Ensure the polygon is closed (first and last vertices are the same)
    if vertices[0] != vertices[-2] or vertices[1] != vertices[-1]:
        vertices.extend(vertices[:2])

    # Calculate the area using the shoelace formula
    area = 0
    n = len(vertices) // 2
    for i in range(n - 1):
        area += (vertices[2*i] * vertices[2*i+3] - vertices[2*i+1] * vertices[2*i+2])
    area += (vertices[2*n-2] * vertices[1] - vertices[2*n-1] * vertices[0])
    return abs(area) / 2

def create_pair(t):
  pair = list(zip(t[::2],t[1::2]))
  new_pair = []
  for i in pair:
    new_pair.append(list(i))
  return new_pair

def remove_small_polygons(mask, threshold):
    if not isinstance(mask,np.ndarray):
        print('Takes a numpy array as input with entries either 0 or 255.')
        return
    segmentation = polygon_vertices(mask)
    poly_return = [polygon for polygon in segmentation if calculate_polygon_area(polygon) > threshold]
    binary_mask = np.zeros(mask.shape, dtype=np.uint8)
    for i in poly_return:
      pair = create_pair(i)
      polygon_points = np.array(pair, np.int32)
      cv2.fillPoly(binary_mask, [polygon_points], 255)
    return binary_mask/255

