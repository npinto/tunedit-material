import numpy as np

np.random.seed(42)

categories = ('A', 'B', 'C')

prelim_labels = [categories[np.random.randint(len(categories))]
                 for _ in xrange(500)]
final_labels = [categories[np.random.randint(len(categories))]
                for _ in xrange(1500)]
lines = [l+'\n' for l in prelim_labels] \
        + ['\n'] + \
        [l+'\n' for l in final_labels]

out_fname = 'random_labels.txt'
print out_fname
open(out_fname, 'w+').writelines(lines)
