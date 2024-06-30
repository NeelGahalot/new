import warnings
import sys
sys.path.append('/teamspace/studios/this_studio/sam-hq')
from segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import box_convert
import os
import supervision as sv
import matplotlib.image as mpimg
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import boto3
import pandas as pd
import argparse
from PIL import Image
import urllib.request

# Ignore SupervisionWarnings SAM_Dino_processing.py
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*annotate is deprecated.*")

# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad=True.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*the name being registered conflicts with an existing name.*")

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options confidence_processed.csv


    parser.add_argument("--sam_path", default='/teamspace/studios/this_studio/sam-hq', type=str,
                        help="Path to SAM Directory")
    parser.add_argument("--dino_path", default='/teamspace/studios/this_studio/GroundingDINO', type=str,
                        help="Path to Grounding Dino Directory")
    parser.add_argument("--dino_config_path", default='/teamspace/studios/this_studio/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', type=str)
    parser.add_argument("--dino_weights_path", default='/teamspace/studios/this_studio/weights/groundingdino_swint_ogc.pth', type=str)
    parser.add_argument("--text_prompt", default='plant', type=str,
                        help="Prompt text for Grounding Dino")
    parser.add_argument("--box_threshold", type=float, default= 0.35,
                        help="Box threshold for grounding dino (default: 0.35)")
    parser.add_argument("--text_threshold", type=float, default= 0.25,
                        help="Text threshold for grounding dino (default: 0.25)")
    parser.add_argument("--sam_upload_threshold", type=float, default= 0.95,
                        help="Mask will be uploaded to s3 if the sam score > this value. (default: 0.95)")
    parser.add_argument("--csv", default='/teamspace/studios/this_studio/india_deeplab_confidence_processed.csv', type=str,
                        help="CSV to read from.")
    parser.add_argument("--bucket", default='treetracker-training-images', type=str,
                        help="S3 bucket to read from.")
    parser.add_argument("--sample_dir", default='india_sam_dino_annotations_large/samples', type=str,
                        help="S3 sample dir to upload samples.")
    parser.add_argument("--mask_dir", default='india_sam_dino_annotations_large/masks', type=str,
                        help="S3 mask dir to upload masks.")
    parser.add_argument("--file_column_name", default='image_url', type=str,
                        help="The name of the columne in the csv that has s3 key/ url of the sample to be downloaded")
    parser.add_argument("--download_from_url", action='store_true', default=True,
                        help="Image may be downloaded from s3 with the s3 obj key or from a URL.")
    parser.add_argument("--output_file_path", default='/teamspace/studios/this_studio/india_sam_dino_itr_1.csv', type=str,
                        help="Path to the output csv.")
    parser.add_argument("--deeplab_threshold", type=float, default= 0.85,
                        help="Mask will be uploaded to s3 if the Deeplab confidence score > this value. (default: 0.95)")
    parser.add_argument("--upload_only", action='store_true', default=False,
                        help="Only uploads the samples and masks to s3 and does not write/ return a csv")
    parser.add_argument("--max_itr", default= None,
                        help='maximum iterations for the main for loop, sometimes we only need limited samples for testing.')
    return parser

def main():
    opts = get_argparser().parse_args()
    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())

    # SAM
    '''
    os.chdir(opts.sam_path)
    #!export PYTHONPATH=$(pwd)
    print(os.getcwd())
    from .segment_anything import sam_model_registry, SamPredictor
    '''

    sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Dino
    os.chdir(opts.dino_path)
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    model = load_model(opts.dino_config_path, opts.dino_weights_path)
    os.chdir('/teamspace/studios/this_studio')
    df_in = pd.read_csv(opts.csv)
    #df_in = df[df['Deeplab Confidence'] > opts.deeplab_threshold]
    s3 = boto3.resource()
    my_bucket = opts.bucket
    
    
    dino_score, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3 = [], [], [], []
    count = 0
    for index, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Processing data"):
        if row['Deeplab Confidence'] < opts.deeplab_threshold:
            value = "Not Processed, low Deeplab Confidence"
            for lst in [dino_score, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
                lst.append(value)
            continue
        if count == opts.max_itr and opts.max_itr != None: 
            break
            
        dino_detected = True
        try:
            if opts.download_from_url:
                url = row[opts.file_column_name]
                local_filename = os.path.join(os.getcwd(),'test.jpg')
                urllib.request.urlretrieve(url, local_filename)
            else:
                s3.Bucket(my_bucket).download_file(row[opts.file_column_name], os.path.join(os.getcwd(),'test.jpg'))
            
            image_source, image = load_image(os.path.join(os.getcwd(),'test.jpg'))
            boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=opts.text_prompt,
      box_threshold=opts.box_threshold,
      text_threshold=opts.text_threshold
  )
            try:
                dino_score.append(logits.numpy()[0])
            except:
                dino_score.append(False)
                dino_detected = False
            if not dino_detected:
                sam_score.append(False)
                sample_uploaded_to_s3.append('No score obtained, no upload attempted')
                mask_uploaded_to_s3.append('No score obtained, no upload attempted')
                continue
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            image = mpimg.imread(os.path.join(os.getcwd(),'test.jpg')).astype(np.uint8)
            predictor.set_image(image)
            input_box = np.array([xyxy[0][0]*image.shape[1], xyxy[0][1]*image.shape[0], xyxy[0][2]*image.shape[1], xyxy[0][3]*image.shape[0]])
            masks, scores, logits = predictor.predict(
      point_coords=None,
      point_labels=None,
      box=input_box[None, :],
      multimask_output=False,
  )
            sam_score.append(scores[0]) 
            #print(scores[0])

            if scores[0] >= opts.sam_upload_threshold:
                
                array = masks[0]
                #print(array.size)
                binary_array = np.where(array, 255, 0).astype(np.uint8)
                binary_image = Image.fromarray(binary_array, 'L')
                #print(os.path.join(os.getcwd(),'test_binarymask.jpg'))
                binary_image.save(os.path.join(os.getcwd(),'test_binarymask.jpg'))
                image_name = 'india_' + row['name'].lower() + os.path.basename(row[opts.file_column_name]).lower()
                
                mask_name = image_name[:-4] + '_binarymask.jpg'
                #sample upload
                s3_sample_filename = os.path.join(opts.sample_dir, image_name)
                #print(s3_sample_filename)
                try:
                    local_sample_name = os.path.join(os.getcwd(),'test.jpg')
                    s3.Bucket(my_bucket).upload_file(local_sample_name, s3_sample_filename)
                    sample_uploaded_to_s3.append(True)
                    count+=1
                    #print('uploaded successfully')
                except Exception as e:
                    raise Exception(f"An error occurred: {str(e)}")
                    print('Sample upload failed')
                    sample_uploaded_to_s3.append('upload attempt failed')

                s3_mask_filename = os.path.join(opts.mask_dir, mask_name)
                try:
                    s3.Bucket(my_bucket).upload_file(os.path.join(os.getcwd(),'test_binarymask.jpg'), s3_mask_filename)
                    mask_uploaded_to_s3.append(True)
                except:
                    print('Mask upload failed')
                    mask_uploaded_to_s3.append('upload attempt failed')
            else:
                sample_uploaded_to_s3.append('Low SAM score, no upload attempted')
                mask_uploaded_to_s3.append('Low SAM score, no upload attempted')
                
        except Exception as e:
            
            pass
            
        for lst in [dino_score, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
            try:
                _ = lst[index]
            except IndexError:
                print('adding none to dino')
                lst.append(None) 
       
    if opts.upload_only:
        print('Called with upload only, upload process concluded.')
        return
    df_in['Grounding Dino Score'] = dino_score
    df_in['SAM Score'] = sam_score
    df_in['Sample Upload'] = sample_uploaded_to_s3
    df_in['Mask Upload'] = mask_uploaded_to_s3

    #output_file_path = '/teamspace/studios/this_studio/sam_dino_processed.csv'
    df_in.to_csv(opts.output_file_path, index=False)
    print('Output CSV file saved successfully.')
    
if __name__ == '__main__':
    main()              


'''        Error

Traceback (most recent call last):
  File "/teamspace/studios/this_studio/SAM_Dino_processing.py", line 156, in <module>
    main()              
  File "/teamspace/studios/this_studio/SAM_Dino_processing.py", line 147, in main
    df_in['SAM Score'] = sam_score
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/frame.py", line 4091, in __setitem__
    self._set_item(key, value)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/frame.py", line 4300, in _set_item
    value, refs = self._sanitize_column(value)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/frame.py", line 5039, in _sanitize_column
    com.require_length_match(value, self.index)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/common.py", line 561, in require_length_match
    raise ValueError(
ValueError: Length of values (107) does not match length of index (155)


Traceback (most recent call last):
  File "/teamspace/studios/this_studio/SAM_Dino_processing.py", line 203, in <module>
    main()
  File "/teamspace/studios/this_studio/SAM_Dino_processing.py", line 193, in main
    df_in['Grounding Dino Score'] = dino_score
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/frame.py", line 4091, in __setitem__
    self._set_item(key, value)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/frame.py", line 4300, in _set_item
    value, refs = self._sanitize_column(value)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/frame.py", line 5039, in _sanitize_column
    com.require_length_match(value, self.index)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/pandas/core/common.py", line 561, in require_length_match
    raise ValueError(
ValueError: Length of values (371) does not match length of index (187)
                
'''
            
    

