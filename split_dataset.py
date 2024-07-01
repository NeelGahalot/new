import os
import random
import shutil
import argparse
import pandas as pd

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.75,
                        help=" train_ratio (default: 0.75)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help=" val_ratio (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.05,
                        help=" test_ratio (default: 0.05)")
    parser.add_argument("--dataset_dir", default='freetown_sam_dino_annotations_large', type=str,
                        help="dataset_dir") 
    parser.add_argument("--from_csv", action='store_true', default=True,
                        help="this csv has metadata about the annotations, e.g, their SAM Score.")
    parser.add_argument("--csv", default='/teamspace/studios/this_studio/pilot_with_crf_large_11769.csv', type=str,
                        help="input CSV to read from.")
    parser.add_argument("--sam_threshold", type=float, default= 0.95,
                        help="Mask will be uploaded to s3 if the sam score > this value. (default: 0.95)")
    return parser

opts = get_argparser().parse_args()

# all datasets are assumed to have 3 directories, namely, samples (for images), binary_masks (for lables), and splits 
image_dir = os.path.join(opts.dataset_dir, 'samples/')
masks_dir = os.path.join(opts.dataset_dir, 'binary_masks/')
output_dir = os.path.join(opts.dataset_dir, 'splits/')

def split_dataset(image_dir, output_dir, train_ratio, val_ratio, test_ratio):
    #images = [img for img in os.listdir(image_dir) if img.lower().endswith('.jpg')]
    if opts.from_csv:
        
        df = pd.read_csv(opts.csv)
        df['SAM Score'] = pd.to_numeric(df['SAM Score'], errors='coerce')
        filtered_list = df[df['SAM Score'] > opts.sam_threshold]['File'].tolist()
        images = []
        for name in filtered_list:
            parts = name.lower().split('/')
            images.append('_'.join(parts))
    else:
        images = [img for img in os.listdir(image_dir)]
        
    

    # Adjust the extension
    total_images = len(images)
    random.shuffle(images)

    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    # Split the dataset
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
      for item in train_images:
        file_name_without_ext = item.lower().rsplit('.jpg', 1)[0]
        f.write("%s\n" % file_name_without_ext)
        #f.write("%s\n" % os.path.splitext(item)[0]) 

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
      for item in val_images:
        file_name_without_ext = item.lower().rsplit('.jpg', 1)[0]
        f.write("%s\n" % file_name_without_ext)
        #f.write("%s\n" % os.path.splitext(item)[0]) 
        
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
      for item in test_images:
        file_name_without_ext = item.lower().rsplit('.jpg', 1)[0]
        f.write("%s\n" % file_name_without_ext)
        #f.write("%s\n" % os.path.splitext(item)[0])  

    return train_images, val_images, test_images






def lowercase_fname(directory):
    for f in os.listdir(directory):
        os.rename(os.path.join(directory, f), os.path.join(directory, f.lower()))
 
 # usage       
split_dataset(image_dir=image_dir, output_dir=output_dir, train_ratio = opts.train_ratio, val_ratio = opts.val_ratio, test_ratio = opts.test_ratio)  

lowercase_fname(image_dir)
lowercase_fname(masks_dir)