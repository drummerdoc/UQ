import cPickle as pickle
f=open('hist.dat','rb')
tmp = pickle.load(f)
f.close()

import matplotlib.pyplot as pl
pl.hist(tmp, 100)
pl.show()

