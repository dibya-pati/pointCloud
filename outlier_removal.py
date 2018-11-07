import pcl
import numpy as np
from glob import glob

indir = r"voxel-cleaned/"
outdir = r"sor-cleaned/"

for infile in glob(indir + '*.xyz'):

    points = []
    outfile = outdir + infile.split('\\')[-1]
    print(infile)

    with open(infile, 'r') as f:
        for line in f:
            x, y, z, _, _, _ = line.strip().split(' ')
            # r ,g ,b = float(r), float(g), float(b)

            # if r < 1 and g < 1 and b < 1:
            #     scale = 255
            # else:
            #     scale = 1

            # r, g, b = int(r*scale), int(g*scale), int(b*scale)
            points.append((float(x), float(y), float(z)))

    points = np.asarray(points, dtype=np.float32)
    print(points.shape)

    p = pcl.PointCloud()
    p.from_array(np.asarray(points, dtype=np.float32))

    # use SOR filter to clean the point cloud
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(20)
    fil.set_std_dev_mul_thresh(1)
    p = fil.filter()

    # use segmentation model to separate two sections in point cloud, uses RANSAC
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(50)
    out_ind, _ = seg.segment()

    out = p.extract(out_ind)

    npout = out.to_array()

    with open(outfile, 'w+') as pcfile:
        for row in npout:
            x, y, z = row
            pcfile.write(str(x) + " " + str(y) + " " + str(z) + "\n")

    print(outfile)
    break