#!/usr/bin/env python3

import argparse
import datetime
import csv
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--csv", required=True, help="Path to Goodreads CSV export")
    parser.add_argument("-o", "--output", required=True, help="Output path")

    args = parser.parse_args()

    # Open as csv.DictReader
    f = open(args.csv, 'r', encoding='utf-8')
    reader = csv.DictReader(f)

    books = {}
    for row in reader:
        if int(row['Read Count']) > 0:
            book = {
                "title": row['Title'],
                "author": row['Author']
            }

            date = row['Date Read']
            if date == "":
                year = "Pre-2014"
            else:
                year = date.split("/")[0]

            print(year)
            if year not in books:
                books[year] = [book]
            else:
                books[year].append(book)


    fw = open(args.output, 'w', encoding='utf-8')
    today = datetime.date.today().strftime("%B %d, %Y")
    fw.write(f"lastupdate: {today}\n")
    fw.write("list:\n")
    for key in books:
        fw.write(f"  - year: {key}\n")
        fw.write(f"    books:\n")
        for book in books[key]:
            fw.write(f"      - title: {book['title']}\n")
            fw.write(f"        author: {book['author']}\n")
            fw.write(f"        star: no\n")

    fw.close()
    f.close()

if __name__ == "__main__":
    main()
