import numpy as np
import sys
from os import path
from glob import glob

def lvmdir_to_npz(dpath, out_fname=None):

    if out_fname is None:
        out_fname = dpath + '_lvm_files.npz'

    if path.exists(out_fname):
        print "%s exists!" % out_fname
        return

    filenames = glob(dpath + '/*.lvm')
    assert len(filenames) > 0
    filenames.sort()
    filenames = np.array(filenames)

    raw_data = []
    print "loading files..."
    for fi, fname in enumerate(filenames):
        print fi + 1, fname
        arr = np.loadtxt(fname, delimiter=',')
        raw_data += [arr]
    print "convert to ndarray..."
    raw_data = np.array(raw_data)

    print "saving", out_fname
    np.savez(out_fname, raw_data=raw_data, filenames=filenames)


def get_data_time_filenames(fname):
    npz = np.load(fname)
    filenames = npz['filenames']
    raw_data = npz['raw_data']
    time = raw_data[:, :, 0]
    data = raw_data[:, :, 1]
    return data, time, filenames


def get_full_data_time_labels_filenames(fname_pattern, categories=('A', 'B', 'C')):
    data = None
    time = None
    filenames = None
    labels = None
    for category in categories:
        fname = fname_pattern % category
        print fname
        d, t, f = get_data_time_filenames(fname)
        l = np.array([category] * len(d))
        if data is None:
            data = d
        else:
            data = np.concatenate([data, d])
        if time is None:
            time = t
        else:
            time = np.concatenate((time, t))
        if filenames is None:
            filenames = f
        else:
            filenames = np.concatenate((filenames, f))
        if labels is None:
            labels = l
        else:
            labels = np.concatenate((labels, l))
    return data, time, labels, filenames


def main():

    if len(sys.argv) != 2:
        print "Usage: python %s <directory_with_lvm_files>" % __file__
        print "Example: python %s ./data/trainFull/trainFullA" % __file__
        sys.exit(1)

    dpath = sys.argv[1]
    out_fname = dpath + '.npz'
    lvmdir_to_npz(dpath, out_fname)

if __name__ == '__main__':
    main()
