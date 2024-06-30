import requests  # total count ~ 6573422
from io import BytesIO
import json
import random
import csv
import argparse
from tqdm import tqdm  # Import tqdm for progress bar

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options confidence_processed.csv Deeplab/pilot_eastafrica_haiti_freetown_WITH_deeplab_crf_sam.csv
    parser.add_argument("--query_url", default='https://prod-k8s.treetracker.org/query/trees?', type=str,
                        help="Path to SAM Directory")
    parser.add_argument("--output_file_path", default='/teamspace/studios/this_studio/big_production_sampling.csv', type=str,
                        help="Path to the output csv.")
    parser.add_argument("--total_count", default=40000, type=int,
                        help='no of samples to get')
    parser.add_argument("--total", default=6573422, type=int,
                        help='total samples in the production database.')
    parser.add_argument("--query_limit", default=100, type=int,
                        help='total samples in the production database, you cannot use a limit>10,000')
    return parser

def main():
    sampled_list = []
    opts = get_argparser().parse_args()
    count = 0
    
    # Initialize the tqdm progress bar
    with tqdm(total=opts.total_count) as pbar:
        while count < opts.total_count:
            try:
                r = random.randint(0, opts.total)
                
                response = requests.get(f"{opts.query_url}offset={r}&limit={opts.query_limit}")
                json_body = json.load(BytesIO(response.content))
                tree_data = json_body.get('trees', [])
                #print(tree_data)
                if tree_data:
                    image_urls = [data.get('image_url') for data in tree_data]
                    increment = len([item for item in image_urls if item not in sampled_list])
                    sampled_list.extend([item for item in image_urls if item not in sampled_list])  
                    
                    count += increment
                    
                    pbar.update(increment)  # Update the progress bar
            except Exception as e:
                print(e)
                continue  # Skip this iteration if there's an error


                

    csv_file = opts.output_file_path

    # Open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_url"])

        # Write each entry as a new row in the CSV
        for entry in sampled_list:
            writer.writerow([entry])

if __name__ == "__main__":
    main()
