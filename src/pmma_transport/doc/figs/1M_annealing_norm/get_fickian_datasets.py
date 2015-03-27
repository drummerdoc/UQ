"""
To get data setup plus some other bits
Other bits will eventually migrate elsewhere
"""

import re
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import emcee
from grab_transport_data import DiffData


import matplotlib.font_manager as font_manager

# Bottom vertical alignment for more space
title_font = {'fontname': 'Arial', 'size': '10',
              'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}

axis_font = {'fontname': 'Arial', 'size': '10'}


plt.close()

simdt = 10
L = 7*2.54e-5
Pa = 101.3325*1000

lnlikecntr = 0
transp_fac = 1.0
bestparm = []
bestfit = -1e99


def get_params_ds(guess, times, transp, sample_id, Ta, RH):

    pDS = np.array(guess)
    global bestfit
    bestfit = -1e99
    global bestparm
    bestparm = []

    bestparm = pDS
    ndim, nwalkers = 4, 8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(times, transp, sample_id, Ta, RH))
    pos = [pDS + pDS*.5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, 500)

    return np.array(bestparm)


def make_label(name, popt):
    return "{}: D={:2.3e}, k={:2.2e}, t0={:4g}".\
           format(name, popt[0], popt[1], popt[2])


def make_label_S(name, popt):
    return "Sim: {}: D={:2.3e}, k={:2.2e}, t0={:4g}".\
           format(name, popt[0], popt[1], popt[2])


def make_label_DS(name, popt):
    return "{}: D={:2.3e}, CHp={:2.2e}, b={:2.2e}, Kd={:2.2e}, alpha=kf/kr={:2.2e}".\
           format(name, popt[0], popt[1], popt[2], popt[3], popt[2]/popt[3])


def make_label_DSF(name, popt):
    return "{}: D={:2.3e}, CHp={:2.2e}, b={:2.2e}, Kd={:2.2e}, alpha=kf/kr={:2.2e}".\
           format(name, popt[0], popt[1], popt[2], popt[3], popt[2]/popt[3])


# Now do fits
times = []
transp = []
fits = []
fitsDS = []
fitsDSF = []
simsols = []
lab = []
labelsDS = []
labelsDSF = []


def spew_data_to_ftn(data):
    f = open("ftn_data.f90", "w")
    f.write("integer, parameter :: nExp={}\n".format(len(data)))
    f.write("integer, dimension({}) :: ndata\n".format(len(data)))
    i = 0
    for D in data:
        i += 1
        f.write("real*8, dimension({}) :: js_{}\n".format(len(D.ts), i))
        f.write("real*8, dimension({}) :: ts_{}\n".format(len(D.js), i))

    i = 0
    for D in data:
        i += 1
        f.write("ndata({}) = {}\n".format(
            i, len(D.ts)))
        f.write("rh_list({}) = {}\n".format(i,
                D.RH))
        f.write("sigma_list({}) = {}\n".format(i,
                D.sigma))
        f.write("temp_list({}) = {}\n".format(i,
                D.temp))
        f.write("thickness_list({}) = {}\n".format(i,
                D.thickness))
        f.write("ts_{} = (/ ".format(i))
        for tt in D.ts[:-1]:
            f.write("{}, &\n".format(tt))
        f.write("{}/)\n".format(D.ts[-1]))
        f.write("js_{} = (/ ".format(i))
        for jj in D.js[:-1]:
            f.write("{}, &\n".format(jj))
        f.write("{}/)\n".format(D.js[-1]))
    f.close()


def get_datasets():
    import glob
    allfiles = glob.glob('./1d_pmma/R*.xlsx')
    files = allfiles # allfiles[0:1]

    files[0] = './1d_pmma/R5590-16-5_5590-16-2_082514.xlsx'
    files[1] = './1d_pmma/R5590-20-3_5590-16-2_082714.xlsx'
    files[2] = './1d_pmma/R5590-20-5_5590-16-2_082814.xlsx'
    files[3] = './1d_pmma/R5590-20-2_5590-16-2_082614.xlsx'
    files = allfiles
    files_txt = "{}".format(files)

    alltimes = []
    alltransp = []
    sample_description = []
    sample_conditions = []
    sample_comments = []
    data = []
    for f in files:
        fname = "{}\n".format(f)

        thisData = DiffData('', f)
        if(re.search(r'pmma', thisData.material, re.I)
                and thisData.thickness == 3 and
                (re.search(r'permatran', thisData.instrument, re.I) or
                    re.search(r'aquatran', thisData.instrument, re.I)) and
                (thisData.RH > 0.96) and thisData.sample_id == '5590-16-2'):
                if(re.search(r'bare', thisData.description, re.I)
                        or re.search('uncoated', thisData.description, re.I)
                        or re.search('none', thisData.description, re.I)):
                    print "--------------------->", fname, thisData.sample_id
                else:
                    continue
        else:
            continue

        data.append(thisData)

        sc = "{};\n T={}, RH={}".format(data[-1].sample_id,
                                        data[-1].temp, data[-1].RH)
        sample_conditions.append(sc)
        sample_description.append(data[-1].description)
        sample_comments.append(data[-1].comments)

        sample_name = "{}".format(f)
        lab.append(sample_name)

    return data


if __name__ == "__main__":
    data = get_datasets()
    spew_data_to_ftn(data)
