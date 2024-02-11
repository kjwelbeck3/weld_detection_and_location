import matplotlib.patches as patches
import numpy as np

bw, bh = 8, 22.5
dx, dy = 2.5, 0
sw, sh = 1.5, 15.06
alpha = 0.5

def add_staple_patch(ax, x, y, angle, cls):
    '''
    R_AB -> Frame of center of bounding box wrt to global A
    R_BC -> Frame of bounding box center rotated wrt to non-rotated
    R_CD -> Frame of rotated straight wrt to center of rotated bounding box
    R_CE -> Frame of top annulus wrt to  center of rotated bounding box
    R_CF -> Frame of bottom annulus wrt to  center of rotated bounding box

    R_Cor1 -> Frame of bounding box origin wrt to bounding box center
    R_Dor2 -> Frame of straight origin wrt to straight center
    '''
    theta = np.radians(angle)
    c,s = np.cos(theta), np.sin(theta)
    T = np.array([  [c,-s, 0], 
                    [s, c, 0],
                    [0, 0, 1]])
    
    T_AB = np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])

    T_BC = T

    T_Cor1 = np.array([ [1, 0, -bw/2],
                        [0, 1, -bh/2],
                        [0, 0, 1]])

    T_CD = np.array([[1, 0, dx],
                     [0, 1, dy],
                     [0, 0, 1]])

    T_Dor2 = np.array([ [1, 0, -sw/2],
                        [0, 1, -sh/2],
                        [0, 0, 1]])

    T_CE = np.array([[1, 0, 0],
                     [0, 1, sh/2],
                     [0, 0, 1]])

    T_CF = np.array([[1, 0, 0],
                     [0, 1, -sh/2],
                     [0, 0, 1]])

    T_Aor1 = T_AB@T_BC@T_Cor1
    T_AE = T_AB@T_BC@T_CE
    T_AF = T_AB@T_BC@T_CF

    if cls == 1:
        T_CD = np.array([[1, 0, -dx],
                        [0, 1, dy],
                        [0, 0, 1]])
    else:
        T_CD = np.array([[1, 0, dx],
                        [0, 1, dy],
                        [0, 0, 1]])
    T_Aor2 = T_AB@T_BC@T_CD@T_Dor2
    
    # markers = ax.scatter(x,y, color="black",marker="+") 
    ax.add_patch(patches.Rectangle((T_Aor1[0,2], T_Aor1[1,2]), bw, bh, angle=angle, ec='red', fc='none', ls='dashed'))
    ax.add_patch(patches.Rectangle((T_Aor2[0,2], T_Aor2[1,2]), sw, sh, angle=angle, ec="red", fc='none', alpha=alpha))
    ax.add_patch(patches.Annulus((T_AE[0,2], T_AE[1,2]), dx+sw/2, sw, ec="red", fc='none', alpha=alpha))
    ax.add_patch(patches.Annulus((T_AF[0,2], T_AF[1,2]), dx+sw/2.0, sw, ec="red", fc='none', alpha=alpha))
    # return markers