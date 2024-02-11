import numpy


def read_point_cloud(path):
    
    fid = open(path, 'rb')
    header = read_ply_header(fid)
    
    # First read to the point of the first offset.
    s1 = header.vertex_size
    s2 = header.y_index - header.x_index
    assert s2 == 4, 'Vertex coordinates not floats'
    assert header.z_index - header.y_index == 4, 'Vertex coordinates not floats'
    if header.x_index > 0:
        fid.read(header.x_index)
    points = numpy.ndarray((header.vertex_ct, 3), numpy.float32, 
                           buffer = fid.read(s1*header.vertex_ct),
                           strides=(header.vertex_size, s2))
    ''' Remove very distant points. '''
    m = ~numpy.any(numpy.abs(points) > 1000.0, axis=1)
    points = points[m, :]
    ''' Keep only nonzero points. '''
    return points[~numpy.all(points == 0.0, axis=1),:]


class ply_header:
    def __init__(self, vertex_ct):
        self.vertex_size = 0
        self.vertex_ct = vertex_ct


def read_ply_header(fid):
    ''' Given a open file handle, parse the header of the PLY file. '''
    type_sizes = {'float': 4, 'double': 8, 'int32': 4, 'uchar32': 4,
                  'uchar': 1, 'ushort': 2, 'short': 2}
    while fid:
        try:
            line = fid.readline().decode('utf-8')
        except UnicodeDecodeError:
            assert False, 'PLY file has invalid header'
        if line.startswith('element vertex'):
            header = ply_header(int(line.split()[2]))
            break
    while fid:
        try:
            line = fid.readline().decode('utf-8')
        except UnicodeDecodeError:
            assert False, 'PLY file has invalid header'
        if line.startswith('element'):
            # Reached the next element type - stop here.
            break
        elif line.startswith('property'):
            p = line.split()
            assert len(p) == 3, 'PLY file has invalid header'
            if p[2] == 'x':
                header.single_precision = p[1] == 'float'
                header.x_index = header.vertex_size
            elif p[2] == 'y':
                header.y_index = header.vertex_size
            elif p[2] == 'z':
                header.z_index = header.vertex_size
            header.vertex_size += type_sizes[p[1]]
        elif line.startswith('end_header'):
            return header
    while fid:
        try:
            line = fid.readline().decode('utf-8')
        except UnicodeDecodeError:
            assert False, 'PLY file has invalid header'
        if line.startswith('end_header'):
            return header

