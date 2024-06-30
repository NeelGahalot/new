import warnings
import sys
from torchvision.ops import box_convert
from torchvision.io import read_image, write_jpeg
from torchvision.ops import masks_to_boxes
import os
from dotenv import load_dotenv
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
import io
from torchvision import transforms
from io import BytesIO

home = os.getcwd()


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options confidence_processed.csv Deeplab/pilot_eastafrica_haiti_freetown_WITH_deeplab_crf_sam.csv


    parser.add_argument("--sam_path", default='/teamspace/studios/this_studio/sam-hq', type=str,
                        help="Path to SAM Directory")
    parser.add_argument("--deeplab_folder_path", default='/teamspace/studios/this_studio/Deeplab', type=str,
                        help="Path to Deeplab Folder")
    parser.add_argument("--cred_path", default='/teamspace/studios/this_studio/Deeplab/credentials.env', type=str,
                        help="Path to boto 3 credentials")

    parser.add_argument("--sam_upload_threshold", type=float, default= 0.92,
                        help="Mask will be uploaded to s3 if the sam score > this value. (default: 0.95)")
    parser.add_argument("--csv", default='/teamspace/studios/this_studio/Deeplab/pilot_11769_eastafrica_haiti_freetown_WITH_deeplab_crf_sam.csv', type=str,
                        help="input CSV to read from.")
    parser.add_argument("--bucket", default='treetracker-training-images', type=str,
                        help="S3 bucket to read from.")
    parser.add_argument("--sample_dir", default='pilot_with_crf_large/samples', type=str,
                        help="S3 sample dir to upload samples.")
    parser.add_argument("--mask_dir", default='pilot_with_crf_large/binary_masks', type=str,
                        help="S3 mask dir to upload masks.")
    parser.add_argument("--file_column_name", default='File', type=str,
                        help="The name of the columne in the csv that has s3 key/ url of the sample to be downloaded")
    parser.add_argument("--output_file_path", default='/teamspace/studios/this_studio/pilot_with_crf_large_11769.csv', type=str,
                        help="Path to the output csv.")
    parser.add_argument("--deeplab_threshold", type=float, default= 0.85,
                        help="Mask will be uploaded to s3 if the Deeplab confidence score > this value. (default: 0.95)")
    parser.add_argument("--upload_only", action='store_true', default=False,
                        help="Only uploads the samples and masks to s3 and does not write/ return a csv")
    parser.add_argument("--max_itr", default= None,
                        help='maximum iterations for the main for loop, sometimes we only need limited samples for testing.')
    parser.add_argument("--ckpt", default='/teamspace/studios/this_studio/Deeplab/saved_models/best_deeplabv3plus_mobilenet_custom_os16_0.7854892764326529.pth',type=str,
                        help="restore from checkpoint")
    return parser

def main():
    opts = get_argparser().parse_args()
    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    load_dotenv(dotenv_path=opts.cred_path)
    
    os.chdir(opts.deeplab_folder_path)
    sys.path.append(os.getcwd())
    from inference.infer import get_s3_bucket, flip, is_s3_object_key, is_url, load_deeplab_model, pil_to_grayscale_tensor
    from post_processing.control_random_field import crf_with_prob
    import utils
    
    
    os.chdir(opts.sam_path)
    sys.path.append(opts.sam_path)
    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    os.chdir(home)
    
    bucket = get_s3_bucket('treetracker-training-images')
    model = load_deeplab_model(opts.ckpt, device).eval()

    

    df_in = pd.read_csv(opts.csv)
    df_in = df_in.sample(frac=1).reset_index(drop=True)
    
    deeplab_confidence, bounding_box_crf, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3 = [], [], [], [], []
    count = 0
    
    for index, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Processing data"):
        if count == opts.max_itr and opts.max_itr != None:
            break
        to_read = row[opts.file_column_name] 
        try:
            if isinstance(to_read, Image.Image):
                image = flip(to_read)
            elif is_url(to_read):
                response = requests.get(to_read)
                image = flip(Image.open(BytesIO(response.content)))
            else:
                s3_object = bucket.Object(to_read).get()
                #print('here')
                image = flip(Image.open(io.BytesIO(s3_object['Body'].read())))
            img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
            img_tensor = img_transform(image).unsqueeze(0).to(device, dtype=torch.float32)
            with torch.no_grad():
                output = model(img_tensor)
                output = torch.squeeze(output, dim=1)
                prob = torch.sigmoid(output).detach()
                pred = (prob > 0.5).long().cpu().numpy()[0]
            denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_np = img_tensor[0].detach().cpu().numpy()
            img_np = (denorm(img_np) * 255).transpose(1, 2, 0).astype(np.uint8)
            prob_np = prob[0].cpu().numpy()
            count_pixel = np.sum(prob_np > 0.5)
            confidence = np.sum(prob_np[prob_np > 0.5]) / count_pixel if count_pixel != 0 else 0
            #print(confidence)
        except Exception as e:
            print("An error occurred:", e)
            value = "Couldn't be processed by Deeplab Checkpoint, something went wrong."
            for lst in [deeplab_confidence, bounding_box_crf, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
                lst.append(value)
            continue
            
        if count_pixel == 0:
            deeplab_confidence.append('Nothing detected')
            value = "Not Processed, nothing detected"
            for lst in [bounding_box_crf, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
                lst.append(value)
            continue
        elif confidence < opts.deeplab_threshold:
            deeplab_confidence.append(confidence)
            value = "Low Confidence."
            for lst in [bounding_box_crf, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
                lst.append(value)
            continue
        else:
            deeplab_confidence.append(confidence)
            cleaned_mask = crf_with_prob(img_np, (pred * 255).astype(np.uint8), prob_np)
            if np.sum(cleaned_mask == 1) == 0:
                value = "Cleaned Mask detects nothing."
                for lst in [bounding_box_crf, sam_score, sample_uploaded_to_s3, mask_uploaded_to_s3]:
                    lst.append(value)
                continue    
            else:
                cleaned_mask_img = Image.fromarray((cleaned_mask * 255).astype(np.uint8))
                bounding_box_crf.append(True)
                boxes = masks_to_boxes(pil_to_grayscale_tensor(cleaned_mask_img))
                box = np.array(boxes.tolist()[0])
                predictor.set_image(img_np)
                masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box[None, :],
    multimask_output=False,
    )
                sam_score.append(scores[0])
                if scores[0] >= opts.sam_upload_threshold:
                    array = masks[0]
                    binary_array = np.where(array, 255, 0).astype(np.uint8)
                    binary_image = Image.fromarray(binary_array, 'L') 
                    # Renaming the files, this might have to change if you are using a url.
                    parts = (row[opts.file_column_name]).lower().split('/')
                    image_name = '_'.join(parts)
                    mask_name = image_name[:-4] + '_binarymask.jpg'
                    
                    s3_sample_filename = os.path.join(opts.sample_dir, image_name)
                    
                    s3_mask_filename = os.path.join(opts.mask_dir, mask_name)
                    try:
                        image_byte_array = io.BytesIO()
                        image.save(image_byte_array, format='JPEG')
                        image_byte_array.seek(0) 
                        bucket.put_object(Key=s3_sample_filename, Body=image_byte_array, ContentType='image/jpeg')
                        #print('uploaded')
                        sample_uploaded_to_s3.append(True)
                        count+=1
                    except:
                        print('Sample upload failed')
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
        
    df_in['Deeplab Confidence'] = deeplab_confidence
    df_in['Bounding Box from CRF'] = bounding_box_crf
    df_in['SAM Score'] = sam_score
    df_in['Sample Upload'] = sample_uploaded_to_s3
    df_in['Mask Upload'] = mask_uploaded_to_s3

    #output_file_path = '/teamspace/studios/this_studio/sam_dino_processed.csv'
    df_in.to_csv(opts.output_file_path, index=False)
    print('Output CSV file saved successfully.')
    
if __name__ == '__main__':
    main()              



                
                
            
                
    
    




