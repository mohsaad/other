#!/usr/bin/env python
# Mohammad Saad
# 6/10/2017
# get_data_from_downloaded_site.py
# A fork of zabihah.py that works completely offline, taking data and
# figuring out things like title and what not

import os
from bs4 import BeautifulSoup
import re
import requests
import json
import sys
import time

def list_all_files(directory):
    files = os.listdir(directory)
    return files

def get_location_data(files, outfile):
    outf = open(outfile, 'w')
    for i in range(0, len(files)):
        f = open("html/" + files[i], 'r')

        soup = BeautifulSoup(f, "html5lib")

        # figure out if this is referring to a mosque or something else
        description = soup.find('title').get_text()
        is_mosque = (description.split(" ")[0].lower() == "mosques")
        if(not is_mosque):
            f.close()
            continue

        # find the titles of each location
        titleBS = soup.find_all('div', attrs={'class': 'titleBS'})
        titles = map(lambda x: x.find('a').get_text(), titleBS)

        # find all addresses
        locs = map(lambda x: x.next_sibling.next_sibling.get_text(), titleBS)

        # write to file / print
        for j in range(0, len(locs)):
            outf.write("{0},{1}\n".format(titles[j].encode('utf-8'), locs[j].encode('utf-8')))

        f.close()

        print("{0}/{1}".format(i,len(files)))

    outf.close()

def geocode_addresses(data_file, api_key_file, outdata):
    api_f = open(api_key_file, 'r')
    api_key = api_f.readline().split("\n")[0]
    api_f.close()

    coordfile = open(outdata, 'w')

    coordinates = []
    test_count = 0
    num_lines = 6907
    start_skip = 0
    end_skip = 0
    start = 0
    end = 1500
    with open(data_file, 'r') as f:
        for i in range(start_skip, end_skip):
            f.readline()

        for i in range(start, end):
            line = f.readline()
            title = line.split(",")[0]
            address_list = line.split('\n')[0].split(',')[1:]
            address = ','.join(address_list)

            url = "https://maps.googleapis.com/maps/api/geocode/json?address={0}&key={1}".format(address.replace(" ", "+"), api_key)
            r = requests.get(url)

            decoded_json = r.json()
            if(len(decoded_json["results"]) == 0):
                continue
            lat = decoded_json["results"][0]["geometry"]["location"]["lat"]
            lon = decoded_json["results"][0]["geometry"]["location"]["lng"]

            address_components = decoded_json["results"][0]["address_components"]
            for i in range(0, len(address_components)):
                if(address_components[i]["long_name"] == "United States"):
                    print("{4}|{0}|{1}|{2}|{3}".format(title, address, lat, lon, i))
                    coordfile.write("{0}|{1}|{2}|{3}\n".format(title, address, lat, lon))


    f.close()
    coordfile.close()


def main():
    #files = list_all_files('html/')
    #get_location_data(files, "data.csv")

    geocode_addresses('data.csv', 'api_key.txt', 'coord_data_3.csv')

if __name__ == '__main__':
    main()
