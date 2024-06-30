import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import boto3
import cv2
import numpy as np
from tqdm import tqdm
#download_thread.py

def resize_image(image_path, output_path):
    img = Image.open(image_path)
    img_resized = img.resize((512, 512))
    img_resized.save(output_path)

def flip_image(image_path, output_path):
    try:
        image = Image.open(image_path)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)

        image.save(output_path)
        image.close()
    except Exception as e:
        pass

def polygon_vertices(mask):
    '''
    Find contours in the binary image and return both external and internal polygons.

    mask: binary image to find contours in
    '''
    _, binary_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    external_polygons = []
    internal_polygons = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Flatten the hierarchy array
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue

            contour = cv2.approxPolyDP(contour, 1, True)
            segmentation = contour.flatten().tolist()

            if hierarchy[i][3] == -1:
                external_polygons.append(segmentation)
            else:
                internal_polygons.append(segmentation)

    return external_polygons, internal_polygons

def calculate_polygon_area(vertices):
    if vertices[0] != vertices[-2] or vertices[1] != vertices[-1]:
        vertices.extend(vertices[:2])

    area = 0
    n = len(vertices) // 2
    for i in range(n - 1):
        area += (vertices[2*i] * vertices[2*i+3] - vertices[2*i+1] * vertices[2*i+2])
    area += (vertices[2*n-2] * vertices[1] - vertices[2*n-1] * vertices[0])
    return abs(area) / 2

def create_pair(t):
    pair = list(zip(t[::2], t[1::2]))
    new_pair = []
    for i in pair:
        new_pair.append(list(i))
    return new_pair

def remove_small_polygons(mask, threshold):
    if not isinstance(mask, np.ndarray):
        print('Takes a numpy array as input with entries either 0 or 255.')
        return
    segmentation, inner_polygons = polygon_vertices(mask)[0], polygon_vertices(mask)[1]

    outer_polygons = [polygon for polygon in segmentation if calculate_polygon_area(polygon) > threshold]

    binary_mask = np.zeros(mask.shape, dtype=np.uint8)
    for i in outer_polygons:
        pair = create_pair(i)
        polygon_points = np.array(pair, np.int32)
        cv2.fillPoly(binary_mask, [polygon_points], 255)

    for i in inner_polygons:
        pair = create_pair(i)
        polygon_points = np.array(pair, np.int32)
        cv2.fillPoly(binary_mask, [polygon_points], 0)

    return binary_mask / 255

def download_and_process_files(bucket_name, s3_directory, local_directory):
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name= os.getenv('AWS_REGION')
    )
    s3 = session.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    def download_file(content):
        try:
            s3_object_key = content['Key']
            folder = args.samples_folder_name if 'samples' in s3_object_key else args.masks_folder_name
            file_name = os.path.basename(s3_object_key)
            local_path = os.path.join(local_directory, folder, file_name)
            s3.download_file(bucket_name, s3_object_key, local_path)
            resize_image(local_path, local_path)

            if folder == args.masks_folder_name:
                mask = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(local_path, (remove_small_polygons(mask, 100)) * 225)
            return local_path
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_directory)
    with ThreadPoolExecutor() as executor:
        futures = []
        for page in response_iterator:
            for content in page.get('Contents', []):
                futures.append(executor.submit(download_file, content))
        for future in tqdm(futures, desc="Downloading files", unit="file"):
            local_path = future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from an S3 bucket, resize images, and flip them based on Exif orientation tag.")
    parser.add_argument("--bucket_name", default='treetracker-training-images', type=str, help="Name of the S3 bucket")
    parser.add_argument("--s3_directory", default='pilot_with_crf_large/', type=str, help="Directory path in the S3 bucket")
    parser.add_argument("--local_directory", default='/teamspace/studios/this_studio/Deeplab/crf_sam_annotations_large', type=str, help="Local directory path to save the downloaded files")
    parser.add_argument("--masks_folder_name", default='binary_masks', type=str, help="Name of the masks folder in the s3 bucket")
    parser.add_argument("--samples_folder_name", default='samples', type=str, help="Name of the samples folder in the s3 bucket")
    
    args = parser.parse_args()

    os.makedirs(os.path.join(args.local_directory, 'binary_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.local_directory, 'samples'), exist_ok=True)

    download_and_process_files(args.bucket_name, args.s3_directory, args.local_directory)
