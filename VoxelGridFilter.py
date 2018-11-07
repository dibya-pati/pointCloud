'''Use pcl (pcl python wrapper) to resample points from point cloud using voxel grid method
voxel grid method creates unifrom voxels in pointcloud, replaces the points in each voxel
with the mean value
cons: if the pc has steep inclination, close to ~90 on the plane, leads to sparse points output
'''

import pcl
import numpy as np
from glob import glob

indir = r"complete-cleaned/"
outdir = r"voxel-cleaned/"

for infile in glob(indir + '*.xyz'):

    points = []
    outfile = outdir + infile.split('\\')[-1]
    print(infile)

    with open(infile,'r') as f:
        for line in f:
            x, y, z, r, g, b = line.strip().split(' ')
            r, g, b = float(r), float(g), float(b)

            if r<1 and g<1 and b<1:
                scale = 255
            else:
                scale = 1

            r, g, b = int(r*scale), int(g*scale), int(b*scale)
            points.append((float(x), float(y), float(z), r << 16 | g << 8 | b ))

    points = np.asarray(points, dtype=np.float32)
    print(points.shape)

    p = pcl.PointCloud_PointXYZRGB()
    p.from_array(np.asarray(points, dtype=np.float32))

    voxelg = p.make_voxel_grid_filter()
    voxelg.set_leaf_size(5, 2.5, 2)
    out = voxelg.filter()

    npout = out.to_array()

    with open(outfile, 'w+') as pcfile:
        for row in npout:
            x, y, z, rgb = row
            rgb = int(rgb)
            r = rgb >> 16 & 0xff
            g = rgb >> 8 & 0xff
            b = rgb & 0xff

            pcfile.write(str(x) + " " + str(y) + " " + str(z) + " "\
                         +str(r) + " " + str(g) + " " + str(b) + "\n")

    print("processed", outfile)
