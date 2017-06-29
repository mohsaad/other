#!/usr/bin/env python
# Mohammad Saad
# 6/14/2017
# draw_us_map.py
# Draws a US map with data points

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

top = 49.3457868 # north lat
left = -124.7844079 # west long
right = -66.9513812 # east long
bottom =  24.7433195 # south lat

map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

# load the shapefile, use the name 'states'
map.readshapefile('shapefiles/st99_d00', name='states', drawbounds=True)

# Read county boundaries
shp_info = map.readshapefile('cb_2016_us_county_500k/cb_2016_us_county_500k', 'counties', drawbounds=True)

lats = []
lons = []

# open up data
f = open('data/coord_data_full.csv', 'r')
count = 0
for line in f:
    count += 1

    info = line.split("\n")[0].split("|")
    if(len(info) != 4):
        break

    print(count)
    lat = float(info[2])
    lon = float(info[3])

    x, y = map(lon, lat)
    map.plot(x,y, 'bo', markersize=0.5)



# x,y = map(lats, lons)
# map.plot(x,y, 'bo', markersize=30)



plt.title('Map of Masjids across America')
# plt.show()
plt.savefig("imgs/masjid_map_counties_low_res.png")
