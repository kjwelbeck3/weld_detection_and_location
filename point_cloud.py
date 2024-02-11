import csv_reader, bin_reader, ply_reader
from math import radians, sin, cos, atan2, pi, sqrt
from scipy.signal import convolve2d
from matplotlib import pyplot
import numpy as np
import open3d as o3d

class point_cloud:
    def __init__(self):
        # Determines how many points are drawn in the point cloud plot.  Does
        # not affect results.  If performance is poor, try reducing this value.
        self.max_draw_points = 200000
        # Initial rotation of point cloud (no rotation).
        self.rotation_matrix = np.identity(3)
        self.path = ''
        self.pc = None
        self.normals = None

    ## Point Cloud loaded in as Nx3 np.array
    def load(self, path):
        ''' Loads a point cloud from input file depending on the file type. '''
        self.path = path

        if path.endswith('csv'):
            self.pc = csv_reader.read_point_cloud(path)
        elif path.endswith('bin'):
            self.pc = bin_reader.read_point_cloud(path)
        elif path.endswith('ply'):
            self.pc = ply_reader.read_point_cloud(path)
        
        # Center of mass must be computed prior to cropping away points since rotation is done
        # first in the weld inspection code.
        self.center_of_mass = np.mean(self.pc, axis=0)
        print("Loaded point cloud: ", self.pc.shape)
        print("File type: ", path.split(".")[-1])

        ## computing normals
        self.compute_normals()

    def get_bounds(self):
        ''' Returns the extents of the bounding box of the point cloud.  '''
        xlo, ylo, zlo = np.min(self.pc, axis=0)
        xhi, yhi, zhi = np.max(self.pc, axis=0)
        return xlo, xhi, ylo, yhi, zlo, zhi

    def draw(self, fig):
        ''' Draws the point cloud to pyplot axis.  '''
        m = np.arange(self.pc.shape[0])
        np.random.shuffle(m)
        m = m[:self.max_draw_points]
        # print(m.shape)
        # print(m[0:-1:10])

        ax = fig.axes[0]
        ax.clear()
        sc = ax.scatter(self.pc[m,0], self.pc[m,1], s=2.0, 
                c=self.pc[m,2], edgecolors='none', cmap='viridis_r')
        if len(fig.axes) == 2:
            fig.axes[1].clear()
            fig.colorbar(sc, ax = ax, cax = fig.axes[1])
        else:
            fig.colorbar(sc, ax = ax)
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)', size=8)
        ax.set_ylabel('y (mm)', size=8)
        ax.grid(axis='both', dashes=(1,1), alpha=0.25, c='k', lw=0.5)
        fig.axes[1].set_ylabel('z (mm)')
        ax.margins(x=0, y=0)

    def compute_normals(self):
        # if self.pc and not self.normals:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pc)
        pcd.estimate_normals()
        self.normals = np.asarray(pcd.normals)
        ## Look out: might have to flip the normal vectors of those that point in the negative z direction
        
    def draw_normals(self, fig):
        m = np.arange(self.pc.shape[0])
        np.random.shuffle(m)
        m = m[:self.max_draw_points]

        ax = fig.axes[0]
        ax.clear()
        sc = ax.scatter(self.pc[m,0], self.pc[m,1], s=2.0, 
                c=self.normals[m,0], edgecolors='none', cmap='viridis_r')
        if len(fig.axes) == 2:
            fig.axes[1].clear()
            fig.colorbar(sc, ax = ax, cax = fig.axes[1])
        else:
            fig.colorbar(sc, ax = ax)
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)', size=8)
        ax.set_ylabel('y (mm)', size=8)
        ax.grid(axis='both', dashes=(1,1), alpha=0.25, c='k', lw=0.5)
        fig.axes[1].set_ylabel('z (mm)')
        ax.margins(x=0, y=0)

            

            

    # ## [!!!: Check the order of multiplictiion in the rotation_matrix update]
    # def align_at_point(self, x, y, rc=3.0):
    #     ''' Calculates the local surface normal direction at the point (x,y)
    #     using all points within radius rc and rotates the point cloud so that
    #     this direction coincides with the z-axis. '''

    #     ## An Nx1 of distances from (x,y)  [Question: Dist should be in 3D not 2D projection]
    #     r2 = (self.pc[:,0] - x)**2 + (self.pc[:,1] - y)**2

    #     ## Index of neighbors ie those below a spec'd threshold + incl a 0.6mm margin
    #     s1 = numpy.nonzero(r2 < (rc+0.6)**2)[0]  ## [0] for the single dim which is nested by default

    #     ## Filter out non-neighbors, find index of then val of closest point
    #     x0 = self.pc[s1[r2[s1].argmin()], :]

    #     ## Vectors from each of neighbors to closest point [Question: Vecs from neighbors to the function params point]
    #     dx = self.pc[s1,0] - x0[0]
    #     dy = self.pc[s1,1] - x0[1]
    #     dz = self.pc[s1,2] - x0[2]

    #     ## Filtering out vectors beyond threshold and arranging into a 3xM
    #     m = (dx**2 + dy**2) < rc*rc
    #     X = numpy.vstack([dx[m], dy[m], dz[m]])

    #     ## Eigenvec with lowest eigenval of the X @ X.T ie non normalized cross-covariance matrix
    #     ## Flips direction of vec if negative
    #     d, v = numpy.linalg.eig(X @ X.T)
    #     e3 = v[:,d.argmin()].T
    #     if e3[2] < 0.0:
    #         e3 *= -1.0

    #     ## Crosses eigen with unit x vector for a second normal
    #     ## Crosses eigen with second normal for a third normal
    #     ## Normalizes each then combines into a Rot Mat
        
    #     e1 = numpy.array([1.0, 0.0, 0.0])
    #     e2 = numpy.cross(e3, e1)
    #     e1 = numpy.cross(e2, e3)
    #     R = numpy.vstack([e1/numpy.linalg.norm(e1),
    #                       e2/numpy.linalg.norm(e2),
    #                       e3])

    #     ## This Rot Matrix represents a new frame in the original frame, 
    #     ## This operation represents all point in this new frame
    #     self.pc = self.pc@R.T# + self.center_of_mass - self.center_of_mass@R.T

    #     ## Updating the frame of the current cloud of points
    #     ## [Question: I think this is misordered!!! POSSIBLE ERROR]  ==> Used to calc euler angles
    #     self.rotation_matrix = R@self.rotation_matrix


    # def rotate_by_degrees(self, x=0.0, y=0.0, z=0.0):
    #     ''' Rotates the point cloud about the x, y, and z axes. '''
    #     x, y, z = [radians(s) for s in [x,y,z]]
    #     Rx = numpy.array([[1.0,    0.0,     0.0],
    #                       [0.0, cos(x), -sin(x)],
    #                       [0.0, sin(x),  cos(x)]])
    #     Ry = numpy.array([[ cos(y), 0.0,  sin(y)],
    #                       [    0.0, 1.0,     0.0],
    #                       [-sin(y), 0.0,  cos(y)]])
    #     Rz = numpy.array([[cos(z), -sin(z),  0.0],
    #                       [sin(z),  cos(z),  0.0],
    #                       [   0.0,     0.0,  1.0]])


    #     R = Rz @ Ry @ Rx

    #     ## Rotating by a combination of axial angles, in the same 
    #     self.pc = self.pc@R.T
    #     # + self.center_of_mass - self.center_of_mass@R.T

    #     ## Updating the frame of the current cloud of points
    #     ## [Question: I am not convinced about this order not about the above order
    #     self.rotation_matrix = R @ self.rotation_matrix


    # def get_euler_angles(self):
    #     R = self.rotation_matrix
    #     qx = 180.0/pi*atan2( R[2,1], R[2,2])
    #     qy = 180.0/pi*atan2(-R[2,0], sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2]))
    #     qz = 180.0/pi*atan2( R[1,0], R[0,0])
    #     return qx, qy, qz



    # def crop(self, xlo=None, xhi=None, ylo=None, yhi=None):
    #     ''' Removes points from point cloud out of bounding box.  '''
    #     m = numpy.full((self.pc.shape[0]), True)
    #     if xlo is not None:
    #         m = numpy.logical_and(m, self.pc[:, 0] > xlo)
    #     if xhi is not None:
    #         m = numpy.logical_and(m, self.pc[:, 0] < xhi)
    #     if ylo is not None:
    #         m = numpy.logical_and(m, self.pc[:, 1] > ylo)
    #     if yhi is not None:
    #         m = numpy.logical_and(m, self.pc[:, 1] < yhi)
    #     self.pc = self.pc[m, :]

    # def compute_normals(self, rcut=4.0):
    #     ''' currently takes too long. '''
    #     bt = bintable.BinTable(self.pc, rcut)
    #     xlo, xhi, ylo, yhi, _, _ = self.get_bounds()
    #     #resolution = 0.2
    #     #nx, ny = int((xhi-xlo)/resolution), int((yhi-ylo)/resolution)
    #     #dx, dy = (xhi-xlo)/nx, 
    #     normals = numpy.zeros(self.pc.shape)
    #     for i in range(self.pc.shape[0]):    
    #         A = numpy.zeros((3,3))
    #         for j in bt.points_near(self.pc[i,:2]):
    #             if i != j:   # [Question:: Is this necessary]
    #                 r = self.pc[j,:] - self.pc[i,:]
    #                 if r.dot(r) > rcut**2:
    #                     continue
    #                 A += numpy.outer(r, r)
    #         d, v = numpy.linalg.eig(A)
    #         n = v[:, numpy.argmin(d)]
    #         if n[2] < 0:
    #             n *= -1.0
    #         normals[i, :] = n
    #     #fig.axes[0].clear()
    #     #fig.axes[0].imagesc(normals, origin='lower')


    # ## Creates a pattern matrix, convolves, finds largest val; no thresholding
    # ## Maps pixels back to point cloud scale; draws dectection overlay
    # ## Averages depth inside and outside hole, then returns difference
    # def detect_hole(self, radius, fig, res=0.2):
    #     ''' Detects the location of a circular hole in the point cloud.
    #     The points must be rotated so that the hole lies in the x-y plane. '''
    #     xlo, xhi, ylo, yhi, zlo, zhi = self.get_bounds()
    #     nx, ny = int((xhi-xlo)/res), int((yhi-ylo)/res)
    #     dx, dy = (xhi-xlo)/nx, (yhi-ylo)/ny
    #     depth, depth_ct = numpy.zeros((nx, ny)), numpy.zeros((nx, ny))
    #     for x in self.pc:
    #         i = min(int((x[0] - xlo) / dx), nx-1)
    #         j = min(int((x[1] - ylo) / dy), ny-1)
    #         depth[i, j] += x[2]
    #         depth_ct[i, j] += 1.0
    #     m = depth_ct > 0.0
    #     depth[m] = depth[m] / depth_ct[m]
    #     d0 = numpy.mean(depth[m])
    #     depth[numpy.logical_not(m)] = d0

    #     # Ratio of the pattern length to the hole diameter.
    #     rf = 1.5 
    #     pnx = int(rf*2.0*radius/res)
    #     pattern = numpy.zeros((pnx, pnx))
    #     xx = numpy.linspace(-rf*radius, rf*radius, pnx)
    #     X, Y = numpy.meshgrid(xx, xx)
    #     R2 = X**2 + Y**2

    #     # m1 is true for within the center of the hole.
    #     m1 = R2 < (radius - 1.0)**2
    #     # m2 is true for points within the inner edge of the hole.
    #     m2 = (R2 < radius**2) & numpy.logical_not(m1)
    #     # m3 is true points outside of the hole.
    #     m3 = numpy.logical_not(m1) & numpy.logical_not(m2)
    #     pattern[m1] = 1.0 / numpy.sum(m1)
    #     pattern[m2] = 0.0
    #     pattern[m3] = -1.0 / numpy.sum(m3)
    #     res = convolve2d(depth, pattern, fillvalue=d0, mode='same')
    #     i, j = numpy.unravel_index(numpy.argmax(res), res.shape)
    #     #fig.axes[0].imshow(res, origin='lower')
    #     xc, yc = xlo + i*dx, ylo + j*dy

    #     pp = dict(lw=1, ec='w', fc='none', ls=':', alpha=0.25)
    #     fig.axes[0].add_patch(pyplot.Circle((xc, yc), radius - 1.0, **pp))
    #     fig.axes[0].add_patch(pyplot.Circle((xc, yc), 1.8*radius, **pp))
    #     fig.axes[0].add_patch(pyplot.Circle((xc, yc), radius, lw=1, ec='limegreen', fc='none'))
    #     fig.axes[0].add_patch(pyplot.Circle((xc, yc), 2.3*radius, **pp))
    #     # hole z-depth
    #     r2 = (self.pc[:,0] - xc)**2 + (self.pc[:,1] - yc)**2
    #     m0 = r2 < (0.7*radius)**2
    #     m1 = (r2 > (1.8*radius)**2) & (r2 < (2.3*radius)**2)
    #     z0 = numpy.mean(self.pc[m0,2])
    #     z1 = numpy.mean(self.pc[m1,2])
    #     return z0-z1 



