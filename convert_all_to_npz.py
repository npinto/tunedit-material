import numpy as np
from os import path
from lvmdir_to_npz import (
    lvmdir_to_npz,
    get_data_time_filenames,
    get_full_data_time_labels_filenames,
)

# ----------------------------------------------------------------------------
print "=" * 80
print "1. convert raw data to intermediate npz files"
print "-" * 80

lvmdirs = [
    # train1000
    './data/train1000/trainA',
    './data/train1000/trainB',
    './data/train1000/trainC',
    # trainFull
    './data/trainFull/trainFullA',
    './data/trainFull/trainFullB',
    './data/trainFull/trainFullC',
    # preliminary
    './data/preliminary',
    # final
    './data/final',
]

for lvmdir in lvmdirs:
    print lvmdir
    lvmdir_to_npz(lvmdir)

# ----------------------------------------------------------------------------
print "=" * 80
print "2. format raw data for 'train*' data"
print "-" * 80

patterns = [
    # train1000
    ('./data/train1000.npz',
     './data/train1000/train%c_lvm_files.npz'),
    # trainFull
    ('./data/trainFull.npz',
    './data/trainFull/trainFull%c_lvm_files.npz'),
]

for out_fname, pattern in patterns:
    print out_fname
    if path.exists(out_fname):
        print "%s already exists" % out_fname
        continue
    data, time, labels, filenames = \
            get_full_data_time_labels_filenames(pattern)
    np.savez(out_fname, data=data, time=time, labels=labels, filenames=filenames)

# ----------------------------------------------------------------------------
print "=" * 80
print "3. format raw data for 'preliminary' data"
print "-" * 80

out_fname = './data/preliminary.npz'
if not path.exists(out_fname):
    print "loading full data"
    full_raw = np.load('./data/trainFull.npz')
    full_data = full_raw['data']
    full_labels = full_raw['labels']

    data, time, filenames = get_data_time_filenames('./data/preliminary_lvm_files.npz')
    labels = []
    for ri, row in enumerate(data):
        print ri+1
        mask = (full_data-row[None, :]).sum(1) == 0
        assert mask.sum() == 1
        label = full_labels[mask]
        labels += [label]
    labels = np.array(labels)

    print out_fname
    np.savez(out_fname, data=data, time=time, labels=labels, filenames=filenames)


# ----------------------------------------------------------------------------
print "=" * 80
print "4. format raw data for 'final' data"
print "-" * 80

out_fname = './data/final.npz'
if not path.exists(out_fname):
    data, time, filenames = get_data_time_filenames('./data/final_lvm_files.npz')
    print out_fname
    np.savez(out_fname, data=data, time=time, filenames=filenames)


