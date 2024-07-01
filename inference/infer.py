from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes, LeafDataset
from torchvision import transforms as T
from metrics import StreamSegMetrics, BinarySegMetrics
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import random
import loralib as lora
from PIL import Image
import urllib.request
import wget
import boto3 
from PIL import Image, ExifTags
import re
import io 
from io import BytesIO
import requests


def get_s3_bucket(bucket_name):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    s3 = boto3.resource(
        's3',
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    return s3.Bucket(bucket_name)

def decode_target(mask):
  leaf_color = [255, 255, 255]
  rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
  rgb_mask[mask == 1] = leaf_color
  return Image.fromarray(rgb_mask)

def is_url(string):
    """
    Check if the given string is a URL.
    """
    # Regular expression to match a URL
    url_regex = re.compile(
        r'^(https?|ftp)://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_regex, string) is not None

def is_s3_object_key(s):
    # Check if the string is an S3 URI
    if s.startswith('s3://'):
        return True
    
    # Check if the string looks like an S3 object key based on your pattern
    # The pattern is more specific to your use case
    if re.match(r'^[^/]+/[^/]+/\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_.+\.jpg$', s):
        return True
    
    return False
    
def show(to_read, my_bucket=None, size=(10, 10)):
    if isinstance(to_read, Image.Image):
        image = flip(to_read)
    elif is_url(to_read):
        response = requests.get(to_read)
        image = flip(Image.open(BytesIO(response.content)))
    elif is_s3_object_key(to_read) and my_bucket is not None:
        s3_object = my_bucket.Object(to_read).get()
        image = flip(Image.open(io.BytesIO(s3_object['Body'].read())))
    else:
        if not os.path.exists(to_read):
            print('Local path not valid.')
            return
        else:
            image = flip(Image.open(to_read))
    
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()

def flip_old(filepath, new_path, do_print = True):
  try:
      image=Image.open(filepath)

      for orientation in ExifTags.TAGS.keys():
          if ExifTags.TAGS[orientation]=='Orientation':

              break
      #print(ExifTags.TAGS.keys())
      exif = image._getexif()
      #print(exif[orientation])

      if exif[orientation] == 3:
          print('Fixing Image Flipping. Exif tag was 3') 
          image=image.rotate(180, expand=True)
      elif exif[orientation] == 6:
          print('Fixing Image Flipping. Exif tag was 6') 
          image=image.rotate(270, expand=True)
      elif exif[orientation] == 8:
          print('Fixing Image Flipping. Exif tag was 8') 
          image=image.rotate(90, expand=True)


      image.save(new_path)
      image.close()
  except:
      # cases: image don't have getexif
      print('no ExifTags found.') 
      pass

def flip(image, do_print=False):
    try:
        # Find the orientation tag in ExifTags
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                if do_print:
                    print('Fixing Image Flipping. Exif tag was 3')
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                if do_print:
                    print('Fixing Image Flipping. Exif tag was 6')
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                if do_print:
                    print('Fixing Image Flipping. Exif tag was 8')
                image = image.rotate(90, expand=True)

        return image
    except AttributeError as e:
        # Cases where the image does not have ExifTags
        if do_print:
            print('No ExifTags found:', e)
        return image  # Return the original image
    except Exception as e:
        # Catch all other exceptions
        if do_print:
            print('An error occurred:', e)
        return image  # Return the original image



def flip_from_path(filepath, new_path, do_print=True):
    try:
        image = Image.open(filepath)

        # Find the orientation tag in ExifTags
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                if do_print:
                    print('Fixing Image Flipping. Exif tag was 3')
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                if do_print:
                    print('Fixing Image Flipping. Exif tag was 6')
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                if do_print:
                    print('Fixing Image Flipping. Exif tag was 8')
                image = image.rotate(90, expand=True)

        image.save(new_path)
        image.close()
    except AttributeError as e:
        # Cases where the image does not have ExifTags
        if do_print:
            print('No ExifTags found:', e)
    except Exception as e:
        # Catch all other exceptions
        if do_print:
            print('An error occurred:', e)
            

img_transform = T.Compose([
              T.Resize((512, 512)),
              T.ToTensor(),
              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])

mask_transform = T.Compose([
              T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST),
              T.ToTensor(),
          ])
    

def get_target(mask_path):
  mask = Image.open(mask_path)
  mask_array = np.array(mask)
  mask_array = (mask_array > 128).astype(np.uint8)
  mask_array = mask_array * 255
  mask = Image.fromarray(mask_array.astype(np.uint8))
  mask = mask_transform(mask)
  mask = torch.squeeze(mask, 0)
  mask = mask.to(device, dtype=torch.long)
  mask = mask.float()
  return mask.cpu().numpy()

def get_exif(image_path):
    try:
        image = Image.open(image_path)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif and orientation in exif:
            print(exif[orientation])

        image.save(output_path)
        image.close()
    except Exception as e:
        print('no tag found or something went wrong')
        pass

def create_histogram(confidence, region, species):

  # list of bins
  bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
  plt.hist(confidence, bins=bins, edgecolor='black')

  if species == None:
    plt.title(region  + ', Image Count - ' + str(len(confidence)))
  else:
    plt.title(region + ' - ' + species + ', Image Count - ' + str(len(confidence)))

  # plotting labelled histogram
  plt.xlabel('Model Confidence')
  plt.ylabel('Image Count')
  plt.show()

import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def load_deeplab_model(ckpt, device, model_type='deeplabv3plus_mobilenet', num_classes=1, output_stride=16):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Device: %s" % device)

    model = network.modeling.__dict__[model_type](num_classes, output_stride)
    network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print("Resume model from %s" % ckpt)
    del checkpoint
    return model

def get_overlayed_mask(checkpoint, to_download, use_url=True, my_bucket = None, save_binary_mask=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_deeplab_model(checkpoint, device).eval()
    img_path = '/teamspace/studios/this_studio/test.jpg'
    if use_url:
        
        urllib.request.urlretrieve(to_download, img_path)
        #!wget -q $to_download -O $img_path
    else:
        my_bucket.download_file(to_download, '/teamspace/studios/this_studio/test.jpg')
        #img_path = to_download  # Assuming `to_download` contains the local path in this case
    flip(img_path,img_path)
    img = Image.open(img_path).convert('RGB')
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = img_transform(img).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.squeeze(output, dim=1)
        prob = torch.sigmoid(output).detach()
        pred = (prob > 0.5).long().cpu().numpy()[0]

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_np = img_tensor[0].detach().cpu().numpy()
    img_np = (denorm(img_np) * 255).transpose(1, 2, 0).astype(np.uint8)

    prob_np = prob[0].cpu().numpy()
    count = np.sum(prob_np > 0.5)
    confidence = np.sum(prob_np[prob_np > 0.5]) / count if count != 0 else 0
    
    if count == 0:
        print('Nothing detected')
    else:
        #print(plt.gcf()) 
        plt.figure(figsize=(10, 10))
        print('Confidence is ' + str(confidence))
        blue_mask = np.zeros_like(img_np)
        blue_mask[:, :, 2] = pred * 255 
        
        plt.imshow(img_np)
        plt.imshow(blue_mask, alpha=0.7)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        overlayed_path = os.path.join(os.getcwd(), 'test_overlayed.jpg')
        
        plt.savefig(overlayed_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
        print('Overlayed Mask saved to ' + os.path.join(os.getcwd(), 'test_overlayed.jpg'))
        #show(overlayed_path)
    
    if save_binary_mask:
        pred_rgb = np.array(decode_target(pred)).astype(np.uint8)
        pred_image = Image.fromarray(pred_rgb).convert('RGB')
        pred_image.save(os.path.join(os.getcwd(), 'test_binary.jpg'))
        print('Binary Mask saved to ' + os.path.join(os.getcwd(), 'test_binary.jpg'))

def pil_to_grayscale_tensor(image):
    grayscale_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    return grayscale_transform(image)




