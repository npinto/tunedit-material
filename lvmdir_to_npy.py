import numpy as np
import sys
from glob import glob

if len(sys.argv) != 2:
    print "Usage: python %s <directory_with_lvm_files>" % __file__
    print "Example: python %s ./data/trainFull/trainFullA" % __file__
    sys.exit(1)

dpath = sys.argv[1]

fname_l = glob(dpath + '/*.lvm')
fname_l.sort()

data = []
print "loading files..."
for fi, fname in enumerate(fname_l):
    print fname, fi+1
    arr = np.loadtxt(fname, delimiter=',')
    data += [arr]
print "convert to ndarray..."
data = np.array(data)

out_fname = dpath + '.npy'
print "saving", out_fname
np.save(open(out_fname, 'wb+'), data)

