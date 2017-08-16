#!/usr/bin/env python
# Mohammad Saad
# 8/15/2017
# get_prayer_times.py
# Gets prayer times off website

import requests
import json
from bs4 import BeautifulSoup
from datetime import date

urlBase = "https://www.islamicfinder.org/world/united-states/5391959/san-francisco-prayer-times/"
class PrayerScraper:

    def __init__(self, url):
        self.url = url
        r = requests.get(self.url)
        self.soup = BeautifulSoup(r.text, 'html5lib')
        self.indices = {}
        self.times = []
        self.year = date.today().year

    def get_webpage(self):
        r = requests.get(self.url)
        self.soup = BeautifulSoup(r.text, 'html5lib')

    def get_prayer_time_table(self):
        prayer_table = self.soup.find(id='monthly-prayers').tbody.find_all('tr')

        # metadata (month)
        metadata = prayer_table[0].find_all('td')
        month = metadata[0].text
        times = []
        indices = {}
        for i in range(0, len(metadata)):
            times.append([])
            indices[metadata[i].text] = i

        for i in range(1, len(prayer_table)):
            if(len(prayer_table[i]['class']) > 1):
                continue
            data = prayer_table[i].find_all('td')
            for j in range(0, len(data)):
                times[j].append(data[j].text)

        self.indices = indices
        self.times = times



def main():
    ps = PrayerScraper(urlBase)
    # ps.get_webpage()
    ps.get_prayer_time_table()

if __name__ == '__main__':
    main()
