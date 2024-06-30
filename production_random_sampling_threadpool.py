import requests
import json
import random
import csv
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_url", default='https://prod-k8s.treetracker.org/query/trees?', type=str,
                        help="Base URL for querying data")
    parser.add_argument("--output_file_path", default='/teamspace/studios/production_sampling_1000_itr_3.csv', type=str,
                        help="Path to the output CSV file")
    parser.add_argument("--total_count", default=4000, type=int,
                        help='Number of samples to get')
    parser.add_argument("--total", default=6573422, type=int,
                        help='Total samples in the production database')
    parser.add_argument("--batch_size", default=100, type=int,
                        help='Batch size for concurrent requests')
    return parser

def fetch_image_urls(offsets, query_url):
    results = []
    for offset in offsets:
        try:
            response = requests.get(f"{query_url}offset={offset}&limit=1")
            json_body = json.loads(response.content)
            tree_data = json_body.get('trees', [])
            if tree_data:
                image_url = tree_data[0].get('image_url')
                results.append(image_url)
        except Exception as e:
            print(f"Error fetching data at offset {offset}: {e}")
    return results

def main():
    opts = get_argparser().parse_args()
    sampled_list = []
    count = 0
    offsets = set()

    with tqdm(total=opts.total_count) as pbar:
        while len(sampled_list) < opts.total_count:
            # Generate a batch of random offsets
            batch_offsets = [random.randint(0, opts.total) for _ in range(opts.batch_size)]
            batch_offsets = [offset for offset in batch_offsets if offset not in offsets]

            # Use ThreadPoolExecutor to fetch image URLs in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for offset in batch_offsets:
                    futures.append(executor.submit(fetch_image_urls, [offset], opts.query_url))
                
                for future in as_completed(futures):
                    results = future.result()
                    if results:
                        sampled_list.extend(results)
                        offsets.update(batch_offsets)
                        pbar.update(len(results))
                        if len(sampled_list) >= opts.total_count:
                            break

            # Clear memory of processed batch_offsets
            del batch_offsets

    # Write the sampled_list to CSV
    csv_file = opts.output_file_path
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_url"])
        for entry in sampled_list:
            writer.writerow([entry])

if __name__ == "__main__":
    main()
