import numpy as np


def read_point_cloud(path):
    fid = open(path, 'rb')
    height = int.from_bytes(fid.read(8), "little", signed = 'true')
    print("height", height)
    length = int.from_bytes(fid.read(8), "little", signed = 'true')
    print("length", length)
    
    # First read to the point of the first offset.
    s1 = length *height
    
    points = np.ndarray((s1, 3), np.double, 
                           buffer = fid.read(24*s1),
                           strides=(24, 8))

    # print(points)                       
    ''' Remove very distant points. '''
    m = ~np.any(np.abs(points) > 1000.0, axis=1)
    points = points[m, :]
    ''' Keep only nonzero points. '''
    points[:,2] = points[:, 2]*(-1)
    return points[~np.all(points == 0.0, axis=1),:]


