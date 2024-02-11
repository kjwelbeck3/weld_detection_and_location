import csv
import numpy as np


def read_point_cloud(path):
    reader = csv.reader(open(path))

    ## Skipping pass header info to the points data
    for row in reader:
        if row and row[0] == 'Y\X':
            break

    ## grouping x,y,z of each point from the file
    ## file is structured as 
    ##      x's as the header row ie horiz axis
    ##      y's as the header col ie vert axis
    ##      z's as the row-col intersections
    
    xx = [float(s) for s in row[1:]]
    xyz = []
    for row in reader:
        if row and row[0] == 'End':
            break
        y = float(row[0])
        zz = [float(s) if s else None for s in row[2:]]
        for i,z in enumerate(zz):
            if z is not None:
                xyz.append([xx[i], y, -z])
    return np.array(xyz)