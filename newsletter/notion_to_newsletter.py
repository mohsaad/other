#!/usr/bin/env python
"""
Script to convert a Notion.so markdown/CSV file
into a newsletter format. This'll only work for a
file with the following format:

Titie | Author | URL | Summary

After formatting, this will be
<url>Title</url>
Author

Summary
"""

import argparse
import csv
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV file to turn into newsletter format", required=True)
    parser.add_argument("--output", help="Output file to store result", required=True)
    args = parser.parse_args()

    f = open(args.csv, 'r')
    newsletter = open(args.output, 'w')

    reader = csv.reader(f)
    start = False
    for line in reader:
        # Skip first line.
        if not start:
            start = True
            continue

        title, author, url, summary = line
        if title == '':
            continue

        article = "{0}\n{1}\n{2}\n\n{3}\n\n".format(title, url, author, summary)
        newsletter.write(article)

    newsletter.close()

if __name__ == '__main__':
    sys.exit(main())
