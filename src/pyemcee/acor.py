#  UQBox code (pronounced like "jukebox" without the "j")

# cheif software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell

#  Compute the auto-covariance function of a time series

import numpy as np

def acor( x, maxLag):
  """return the auto-correlation of x"""

#  input x:      a 1D numpy array whose auto-covariance you seek
#                get n = len(x), use all of x
#  input maxLag: compute C[0], ..., C[maxLag-1]

#  output: C[t] =~ corr(x(k),x(k+t)) = cov(x(k),x(k+t))/var(x) , for t = 0, ..., maxLag-1
#  complain if maxLag > n/2.

  n = len(x)
  if n < 2*maxLag:
    print "auto-covariance function acov has n = " + str(n) + ", and maxLag = " + str(maxLag)
    return []

  C = np.zeros(maxLag)
  for t in range(0,maxLag):
    cm   = np.cov(x[0:n-t], x[t:n])  # The numpy covariance function computes a 2X2 covariance matrix
    C[t] = cm[0,1]

  C = C/C[0]
  return C
