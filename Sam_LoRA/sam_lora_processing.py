import torch
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import yaml
import torch.nn.functional as F
import sys
import os
import argparse
from dotenv import load_dotenv
import boto3
import pandas as pd
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import urllib.request
import io
from urllib.parse import urlparse

home = os.getcwd()

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options confidence_processed.csv Deeplab/pilot_eastafrica_haiti_freetown_WITH_deeplab_crf_sam.csv


    parser.add_argument("--sam_yaml_path", default='/teamspace/studios/this_studio/Sam_LoRA/config.yaml', type=str,
                        help="Path to SAM yaml file")
    parser.add_argument("--sam_folder_path", default='/teamspace/studios/this_studio/Sam_LoRA', type=str,
                        help="Path to SAM lora directory")
    parser.add_argument("--sam_upload_threshold", type=float, default= 0.98,
                        help="Mask will be uploaded to s3 if the sam score > this value. (default: 0.95)")
    parser.add_argument("--csv", default='/teamspace/studios/this_studio/big_production_sampling.csv', type=str,
                        help="input CSV to read from.")
    parser.add_argument("--sample_dir", default='production_large_40000_lora_sam_annotations/samples', type=str,
                        help="S3 sample dir to upload samples.")
    parser.add_argument("--mask_dir", default='production_large_40000_lora_sam_annotations/binary_masks', type=str,
                        help="S3 mask dir to upload masks.")
    parser.add_argument("--file_column_name", default='image_url', type=str,
                        help="The name of the columne in the csv that has s3 key/ url of the sample to be downloaded")
    parser.add_argument("--output_file_path", default='/teamspace/studios/this_studio/production_40000_large_annotations_sam_lora.csv', type=str,
                        help="Path to the output csv.")
    parser.add_argument("--cred_path", default='/teamspace/studios/this_studio/Deeplab/credentials.env', type=str,
                        help="Path to boto 3 credentials")
    parser.add_argument("--upload_only", action='store_true', default=True,
                        help="Only uploads the samples and masks to s3 and does not write/ return a csv")
    parser.add_argument("--max_itr", default= 500,
                        help='maximum iterations for the main for loop, sometimes we only need limited samples for testing.')
    parser.add_argument("--deeplab_folder_path", default='/teamspace/studios/this_studio/Deeplab', type=str,
                        help="Path to Deeplab Folder")
    parser.add_argument("--ckpt", default='/teamspace/studios/this_studio/Deeplab/saved_models/best_deeplabv3plus_mobilenet_custom_os16_0.7854892764326529.pth',type=str,
                        help="restore from checkpoint")
    return parser

def main():
    opts = get_argparser().parse_args()
    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
    os.chdir(opts.deeplab_folder_path)
    sys.path.append(os.getcwd())
    from inference.infer import get_s3_bucket, flip, is_s3_object_key, is_url, load_deeplab_model, pil_to_grayscale_tensor
    from post_processing.control_random_field import crf_with_prob
    load_dotenv(dotenv_path=opts.cred_path)
    sys.path.append(opts.sam_folder_path)
    import src.utils as utils
    from src.dataloader import DatasetSegmentation, collate_fn
    from src.processor import Samprocessor
    from src.segment_anything import build_sam_vit_b, SamPredictor
    from src.lora import LoRA_sam

    with open(opts.sam_yaml_path, "r") as ymlfile:
       config_file = yaml.load(ymlfile, Loader=yaml.Loader)
    
    lora = torch.load(config_file["SAM"]["CHECKPOINT"])
    processor = Samprocessor(lora.sam)
    predictor = SamPredictor(lora.sam)
    print(predictor)
    
    os.chdir(home)
    
    bucket = get_s3_bucket('treetracker-training-images')
    
    df_in = pd.read_csv(opts.csv)
    df_in = df_in.sample(frac=1).reset_index(drop=True)

    sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3 = [], [], []
    count = 0
    for index, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Processing data"):
        if count == opts.max_itr and opts.max_itr != None:
            break
        to_read = row[opts.file_column_name] 
        try:
            response = requests.get(to_read)
            image = flip(Image.open(BytesIO(response.content)))
        except Exception as e:
            print("An error occurred:", e)
            value = "Image not read from url"
            for lst in [sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
                lst.append(value)
            continue
        
        image = np.array(image)

        predictor.set_image(image)
        input_box = np.array([0,0,image.shape[0],image.shape[1]])
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        sam_score.append(scores[0])
        if scores[0] >= opts.sam_upload_threshold:
            array = masks[0]
            binary_array = np.where(array, 255, 0).astype(np.uint8)
            binary_image = Image.fromarray(binary_array, 'L')
            # Renaming the files, this might have to change if you are using a url.
            parts = (row[opts.file_column_name]).lower().split('/')
            #filename = urlparse(row[opts.file_column_name]) replace("https://", "")
            image_name = 'production_training_' + str(count) + '.jpg'
            mask_name = image_name[:-4] + '_binarymask.jpg'
        
            s3_sample_filename = os.path.join(opts.sample_dir, image_name)
            
        
            s3_mask_filename = os.path.join(opts.mask_dir, mask_name)
            try:
                image_byte_array = io.BytesIO()
                image = Image.fromarray(image)
                image.save(image_byte_array, format='JPEG')
                image_byte_array.seek(0)
                bucket.put_object(Key=s3_sample_filename, Body=image_byte_array, ContentType='image/jpeg')
                #print('uploaded')
                sample_uploaded_to_s3.append(True)
                count += 1
            except Exception as e:
                print("An error occurred:", e)
                print('Sample upload failed')
                #return
                sample_uploaded_to_s3.append('upload attempt failed')
            try:
                image_byte_array = io.BytesIO()
                binary_image.save(image_byte_array, format='JPEG')
                image_byte_array.seek(0)
                bucket.put_object(Key=s3_mask_filename, Body=image_byte_array, ContentType='image/jpeg')
                mask_uploaded_to_s3.append(True)
            except:
                print('Mask upload failed')
                mask_uploaded_to_s3.append('upload attempt failed')
        else:
            sample_uploaded_to_s3.append('Low SAM score, no upload attempted')
            mask_uploaded_to_s3.append('Low SAM score, no upload attempted')
            
    if opts.upload_only:
        print('Called with upload only, upload process concluded.')
        return
        
    df_in['SAM Score'] = sam_score
    df_in['Sample Upload'] = sample_uploaded_to_s3
    df_in['Mask Upload'] = mask_uploaded_to_s3
    
    df_in.to_csv(opts.output_file_path, index=False)
    print('Output CSV file saved successfully.')
    
if __name__ == '__main__':
    main()              








