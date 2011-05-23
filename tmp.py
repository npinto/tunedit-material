import numpy as np
import sys
from matplotlib import mlab
from proxy import MultiOVALinearLibSVMScikits
from scipy import ndimage

PSTRIDE = 1
FSTRIDE = 10
EPS = 1e-6
C = 1#e-5
MIN = 1.17e-7
MAX = 5.33e-6
SPHERE = True
TRAIN_SPLIT = True

#from joblib import Memory
#mem = Memory('./_cache', verbose=False)
#@mem.cache
def transform_one(row):
    row -= MIN
    row /= MAX
    spec = mlab.specgram(row, NFFT=256, Fs=16384)[0]
    spec = np.log(spec)#.ravel()
    #spec = ndimage.gaussian_filter(spec, 1)
    return spec

def transform(data):
    spec_l = []
    len_data = len(data)
    for ri, row in enumerate(data):
        sys.stdout.write("transform: %.01f%%\r" % (100.*(ri+1)/len_data))
        sys.stdout.flush()
        spec = transform_one(row)
        spec_l += [spec]
    print
    spec = np.array(spec_l)
    return spec

# -- TRAINING
print "training: load data"
trn_raw = np.load('./data/train1000.npz')

if TRAIN_SPLIT:
    trn_data = trn_raw['data'][::2, ::FSTRIDE]
    trn_labels = trn_raw['labels'][::2]
else:
    trn_data = trn_raw['data'][:, ::FSTRIDE]
    trn_labels = trn_raw['labels']

trn_data = trn_data[::PSTRIDE]
trn_data = transform(trn_data)
assert len(trn_data) == len(trn_labels)

print "training: clf"
clf = MultiOVALinearLibSVMScikits(C=C, sphere=SPHERE)
clf.fit(trn_data, trn_labels)

gv = clf.predict(trn_data)
gt = trn_labels
print 'training: perf', (gv==gt).mean()

# -- TEST
print "testing: load data"
tst_raw = np.load('./data/preliminary.npz')

if TRAIN_SPLIT:
    tst_data = trn_raw['data'][1::2, ::FSTRIDE]
    tst_labels = trn_raw['labels'][1::2]
else:
    tst_data = tst_raw['data'][:, ::FSTRIDE]
    tst_labels = tst_raw['labels']

tst_data = tst_data[::PSTRIDE]
tst_data = transform(tst_data)

assert len(tst_data) == len(tst_labels)

print "testing: clf"
gv = clf.predict(tst_data)

gt = tst_labels
print 'testing: perf', (gv==gt).mean()




