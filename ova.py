

import cPickle as pkl

import time
import numpy as np
import scipy as sp

# --
DEFAULT_C = 1e5
DEFAULT_SPHERE = True

# -- SCIKITS
import scikits.learn.svm as svm
class MultiOVALinearLibSVMScikits(object):

    def __init__(self, C=DEFAULT_C, sphere=DEFAULT_SPHERE):
        """XXX: docstring for __init__"""

        self.C = C
        self.sphere = sphere

    def fit(self, data, lbls):
        """XXX: docstring for fit"""

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        categories = np.unique(lbls)
        assert categories.size > 2

        ntrain = len(lbls)

        assert data.shape[0] == ntrain

        data = data.copy()

        data.shape = ntrain, -1

        if self.sphere:
            print ">>> Computing normalization vectors"
            start = time.time()
            fmean = data.mean(0)
            fstd = data.std(0)
            np.putmask(fstd, fstd==0, 1)
            end = time.time()
            print "Time: %s" % (end-start)

            print ">>> Normalizing training data"
            # XXX: use scikits...Scaler
            start = time.time()
            data -= fmean
            data /= fstd
            end = time.time()
            print "Time: %s" % (end-start)

            assert not np.isnan(data).any()
            assert not np.isinf(data).any()

        print ">>> Computing traintrain linear kernel"
        start = time.time()
        print data.shape
        kernel_traintrain = np.dot(data, data.T)
        ktrace = kernel_traintrain.trace()
        ktrace = ktrace != 0 and ktrace or 1
        kernel_traintrain /= ktrace
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Train LibSVMs (C=%e)" % self.C
        start = time.time()

        cat_index = {}

        alphas = {}
        support_vectors = {}
        biases = {}
        clfs = {}

        lbls = np.array(lbls)
        for icat, cat in enumerate(categories):
            print "> [%d] positive label: '%s'" % (icat+1, cat)
            ltrain = np.zeros(len(lbls))
            ltrain[lbls != cat] = -1
            ltrain[lbls == cat] = +1
            assert np.unique(ltrain).size == 2

            clf = svm.SVC(kernel='precomputed', C=self.C)
            clf.fit(kernel_traintrain, ltrain)

            alphas[cat] = clf.dual_coef_
            support_vectors[cat] = clf.support_
            biases[cat] = clf.intercept_
            cat_index[cat] = icat
            clfs[cat] = clf

        end = time.time()
        print "Time: %s" % (end-start)

        self._train_data = data
        self._ktrace = ktrace
        if self.sphere:
            self._fmean = fmean
            self._fstd = fstd
        self._support_vectors = support_vectors
        self._alphas = alphas
        self._biases = biases
        self._clfs = clfs

        self.categories = categories

    def transform(self, data):

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        ntest = len(data)

        data = data.copy()

        data.shape = ntest, -1

        if self.sphere:
            print ">>> Normalizing testing data"
            start = time.time()
            data -= self._fmean
            data /= self._fstd
            end = time.time()
            print "Time: %s" % (end-start)

            assert not np.isnan(data).any()
            assert not np.isinf(data).any()

        print ">>> Computing traintest linear kernel"
        start = time.time()
        kernel_traintest = np.dot(self._train_data, data.T)

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        kernel_traintest /= self._ktrace

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Collecting %d testing outputs" % ntest
        start = time.time()
        categories = self.categories
        support_vectors = self._support_vectors
        alphas = self._alphas
        biases = self._biases
        clfs = self._clfs

        outputs = np.zeros((ntest, len(categories)), dtype='float32')

        print "Predicting testing data ..."
        for icat, cat in enumerate(categories):
            #index_sv = support_vectors[cat]
            #resps = np.dot(alphas[cat],
                           #kernel_traintest[index_sv]) + biases[cat]
            clf = clfs[cat]
            resps = clf.decision_function(kernel_traintest.T)
            resps = resps.ravel()
            outputs[:, icat] = resps

        end = time.time()
        print "Time: %s" % (end-start)

        return outputs

    def predict(self, data):
        """XXX: docstring for transform"""

        cats = self.categories

        outputs = self.transform(data)
        preds = outputs.argmax(1)
        lbls = [cats[pred] for pred in preds]
        return lbls

def main():

    fname = 'ova.pkl'
    data = pkl.load(open(fname))

    trn_feats = data['trn_feats']
    trn_lbls = data['trn_lbls']
    tst_feats = data['tst_feats']
    tst_lbls = data['tst_lbls']

    trn_feats.shape = trn_feats.shape[0], -1
    tst_feats.shape = tst_feats.shape[0], -1

    print trn_feats.shape

    #clf = MultiOVALinearLibSVMShogun(C=1e5)
    clf = MultiOVALinearLibSVMScikits(C=1e5)
    clf.fit(trn_feats, trn_lbls)
    gv = clf.predict(tst_feats)

    #gt = [int(elt[-1]) for elt in tst_lbls]
    gt = tst_lbls

    print len(gv)
    print len(gt)

    print (gv==gt).mean()

# ---
if __name__ == '__main__':
    main()


