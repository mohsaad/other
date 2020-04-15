#!/usr/bin/env python3

import argparse
import requests
import shutil

DEFAULT_URL = "https://eol.jsc.nasa.gov/DatabaseImages/ESC/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Mission Name")
    parser.add_argument("--start", required=True, help="Start index", type=int)
    parser.add_argument("--end", required=True, help="End index", type=int)
    parser.add_argument("--size", default='small', choices=['small', 'large'], help="Size of image")
    parser.add_argument("--output_dir", required=True, help="Output folder")

    # Parse arguments.
    args = parser.parse_args()

    for idx in range(args.start, args.end):
        # Build url.
        url = '{0}/{1}/{2}/{2}-E-{3}.JPG'.format(DEFAULT_URL, args.size, args.mission, idx)
        output_file = "{}/{}.JPG".format(args.output_dir, idx)

        r = requests.get(url, stream=True)
        if r.status_code == 200:
            print("Saving image {} to {}".format(idx, output_file))
            with open(output_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

if __name__ == '__main__':
    main()
