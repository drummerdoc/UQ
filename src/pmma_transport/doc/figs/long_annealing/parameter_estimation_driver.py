"""
Drive parameter estimation using emcee
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import linregress
from sparse_pdf_check import *

import emcee
import triangle

import get_fickian_datasets

from analy_fcn import wvtr_vec, wvtr_vec_global
import sparse_pdf

from is_tools import *

from scipy.cluster.vq import kmeans2

# Load experimental data
print "Scanning data files for experiments matching criteria"
data = get_fickian_datasets.get_datasets()


# Error function for global fitting
def esq_global(par, data):
    L = 0.0
    for i in range(len(data)):
        fsim = wvtr_vec_global(data[i].ts, data[i].RH, data[i].temp,
                               data[i].thickness, *par)
        L = L + np.sum((fsim - data[i].js)**2)
    return L*1e12  # Scale so that it is O(1)


# Likelihood function for emcee
# Gives back (positive) log likelihood
def lnlike(par, data):
    L = 0.0
    for i in range(len(data)):
        fsim = wvtr_vec_global(data[i].ts, data[i].RH, data[i].temp,
                               data[i].thickness, *par)
        inv_sigma2 = 1/(data[i].sigma * data[i].sigma)
        L = L - 0.5 * np.sum((fsim - data[i].js)**2 * inv_sigma2)
    return L


# MLE fitting - single
def get_params_MLE(times, transp, RH, Ta, thickness):
    parms = (2.0e-9*1e-4, 9.0e-3/1e-4, 1.0)
    popt3, pcov3 = curve_fit(lambda times, *p0: wvtr_vec(times, RH,
                             Ta, thickness, *p0), times, transp, p0=parms)
    return popt3


# MLE fitting - global
def get_params_global(data):

    # First curve fit a single experiment to get a decent
    # initial guess for the global fit
    # parms = (2.0e-9*1e-4, 9.0e-3/1e-4, 1.0, 1.0)
    parms = 1e-7, 1e-3, -1000, 1000
    i = 1
    pg, pcovg = curve_fit(lambda times, *p0: wvtr_vec_global(data[i].ts,
                          data[i].RH, data[i].temp, data[i].thickness, *p0),
                          data[i].ts, data[i].js,
                          p0=parms)

    # This is global fit using function above
    eg = lambda par: esq_global(par, data)
    pg2 = minimize(eg, pg, method='Nelder-Mead', tol=1.0e-1,
                   options={"maxfev": 10000, "maxiter": 10000})

    print "Global fitting residual: ", eg(pg2.x)
    return pg2.x


def broad_prior():
    """ Define what a broad prior is
    Trying to make it somewhat reasonable in the sense that
    we do not need to solve an optimization problem to get it
    but yet it is not so naive as to be unbelievable.

    Try correct sign, order of magnitude input
    Let prior span 2 order of magnitude
    ... even range_scale = 2.0 seems to not work so well
    """

    magnitude = [1e-7, 1e-3, 1e3, 1e3]
    sign = [1, 1, -1, 1]
    range_scale = 10.0
    lower_bounds = np.zeros_like(magnitude)
    upper_bounds = np.zeros_like(magnitude)
    i = 0
    for (mag, sgn) in zip(magnitude, sign):
        if(sgn < 0):
            lower_bounds[i] = mag*range_scale*sgn
            upper_bounds[i] = mag/range_scale*sgn
        else:
            lower_bounds[i] = mag/range_scale
            upper_bounds[i] = mag*range_scale
        i += 1

    return np.array(lower_bounds), np.array(upper_bounds)


def get_walkers(N, bounds_fcn, samples=None, data=None):
    walkers = []
    if(not (samples == None)):
        # Get a bunch of samples at random
        cand_samp_idx = np.random.randint(low=0, high=samples.shape[0], size=50000)
        cand_samp = samples[cand_samp_idx]
        csps = []
        for cs in cand_samp:
            csp = lnlike(cs, data)
            csps.append(csp)
        # Then sort by lnlike
        srt_idx = range(len(csps))
        srt_idx.sort(key=csps.__getitem__, reverse=True)
        srt_samp = map(cand_samp.__getitem__, srt_idx)
        for w in range(N):
            idx = np.random.randint(low=0, high=len(srt_samp)/100)
            walkers.append(samples[idx])
        i = 0
        for w in walkers:
            pp = lnlike(w, data)
            print "Walker ", i, " at ", w, " lnlike = ", pp
            i += 1
    else:
        lb, ub = bounds_fcn()
        print "D0", lb[0],  ub[0]
        print "K0", lb[1],  ub[1]
        print "D1", lb[2],  ub[2]
        print "K1", lb[3],  ub[3]
        for i in range(N):
            this_walker = []
            for l, u in zip(lb, ub):
                t = np.random.uniform()
                this_walker.append(l + t*(u-l))
            walkers.append(np.array(this_walker))
    return walkers

def get_walkers_cluster(N, bounds_fcn, samples=None, data=None):
    walkers = []
    if(not (samples == None)):
        # Get a bunch of samples at random
        cand_samp_idx = np.random.randint(low=0, high=samples.shape[0], size=50000)
        cand_samp = samples[cand_samp_idx]

        # Cluster this subset of samples and evaluate likelihood
        k = N + 2
        cents, lab = kmeans2(cand_samp, k)
        csps = []
        for cs in cents:
            csp = lnlike(cs, data)
            csps.append(csp)

        # Then sort by lnlike
        srt_idx = range(len(csps))
        srt_idx.sort(key=csps.__getitem__, reverse=True)
        srt_samp = map(cents.__getitem__, srt_idx)

        # Keep first N (implicitly, dropping N-k clusters)
        walkers = srt_samp[0:N]
        i = 0
        for w in walkers:
            pp = lnlike(w, data)
            print "Walker ", i, " at ", w, " lnlike = ", pp
            i += 1
    else:
        lb, ub = bounds_fcn()
        print "D0", lb[0],  ub[0]
        print "K0", lb[1],  ub[1]
        print "D1", lb[2],  ub[2]
        print "K1", lb[3],  ub[3]
        for i in range(N):
            this_walker = []
            for l, u in zip(lb, ub):
                t = np.random.uniform()
                this_walker.append(l + t*(u-l))
            walkers.append(np.array(this_walker))
    return walkers


def lnprior(par, pg, sparsepdf=None, bf=None):
    if sparsepdf:
        a = spdf.getProbability(par)
        if(a > 1e-30):
            return np.log(spdf.getProbability(par))
        else:
            return -np.inf

    else:
        D0, C0, D1, C1 = par
        if(bf):
            lb, ub = bf()
            if(lb[0] < D0 < ub[0] and
                    lb[1] < C0 < ub[1] and
                    lb[2] < D1 < ub[2] and
                    lb[3] < C1 < ub[3]):
                return 0.0
            else:
                return -np.inf
        else:
            # Bug. shoud not get here.
            sys.exit(-1)
            D0, C0, D1, C1 = par
            a = 0.02095132
            fac = 0.99
            if(pg[0]*(1-fac) < D0 < pg[0]*(1+fac)
                    and pg[1]*(1-fac) < C0 < pg[1]*(1+fac)
                    and pg[2]*(1-fac) > D1 > pg[2]*(1+fac)
                    and pg[3]*(1-fac) < C1 < pg[3]*(1+fac)):
                return 0.0
            return -np.inf


def lnprob(par, data, pg, spdf=None, bf=None):
    if(spdf):
        Lp = lnprior(par, pg, spdf, bf)
    else:
        Lp = lnprior(par, pg, None, bf)
    if not np.isfinite(Lp):
        return -np.inf
    return (Lp + lnlike(par, data))


def get_log_fits(x, y):
    logy = np.log(np.array(y))
    m, b, r, p, se = linregress(x, logy)
    print "m, b, r-value: ", m, b, r
    return np.exp(b), m


if __name__ == "__main__":
    # First MLE fitting stuff - sampling below --------------------------

    # Hack data so that experiment 0 and 1 has larger variance than normal
    data = copy.deepcopy(data[0:2])
    data[0].sigma = data[0].sigma  # *10
    data[1].sigma = data[1].sigma  # *10

    # Plot out raw data along with error bars and MLE results
    nplots = len(data)
    ncols = 2
    if(nplots % ncols != 0):
        nrows = int(nplots/ncols) + 1
    else:
        nrows = nplots/ncols

    pfits = []
    temps = []

    # First pass - do fits for D, k for each experiment independently
    print "Fitting each experiment for D, K independently"
    for i in range(len(data)):
        P3 = (2.0e-9*1e-4, 9.0e-3/1e-4, 1.0, 1.0)
        p3 = get_params_MLE(data[i].ts, data[i].js, data[i].RH, data[i].temp,
                            data[i].thickness)
        print "  D, K for T=", data[i].temp, ", RH=", data[i].RH, ":", p3[0:2]
        pfits.append(p3)
        temps.append(data[i].temp)

    # Do global regression on individual fits
    print "Computing fit of fits:"
    print "(global parameters from fitting individual D, K"
    print " to D=D_0 exp(D_1/T), K=K_0exp(K_1/T) )"
    print " and global fits, from single minimization"
    pfits = np.array(pfits)
    invtemps = 1.0/(273.15+np.array(temps))
    D0, D1 = get_log_fits(invtemps, pfits[:, 0])
    K0, K1 = get_log_fits(invtemps, pfits[:, 1])
    pg_2step = [D0, K0, D1, K1]
    print "     Parameters from 2 step global fitting (D0, K0, D1, K1)=\n",\
          pg_2step

    # Do global fit on all data
    pg_1step = get_params_global(data)
    print "     Parameters from 1 step global fitting (D0, K0, D1, K1)=\n",\
          pg_1step

    # Likelihood of fits
    p1 = lnlike(pg_1step, data)
    p2 = lnlike(pg_2step, data)
    print "   Likelihood for fits: 2step=", p2, " 1step=", p1

    # Plot the two fitting options
    ffit, axfit = plt.subplots(1, 1)
    axfit.plot(np.array(np.array(temps)+273.15), pfits[:, 0], 'bx')
    axfit.set_xlabel("T [K]")
    axfit.set_ylabel("D")
    plttemps = np.linspace(280, 315, 50)
    axfit.plot(plttemps, pg_2step[0]*np.exp(pg_2step[2]/plttemps),
               '--g', lw=2, label="Sequential fitting")
    axfit.plot(plttemps, pg_1step[0]*np.exp(pg_1step[2]/plttemps),
               '--r', lw=2, label="Global fitting")
    plt.legend(loc=0)
    plt.savefig("seq_vs_global_ls.png")
    plt.close()

    # Plot parameters in global fit consistent with each experiment
    def D0vsD1(invT, D, D1):
        return D / np.exp(D1*invT)
    f2, f2a1 = plt.subplots(1, 1)
    D1grid = np.linspace(-4600, -3200, 10)
    for invT, D in zip(invtemps, pfits[:, 0]):
        D0vals = D0vsD1(invT, D, D1grid)
        print invT, D,  "; D0vals:", D0vals
        f2a1.plot(D1grid, D0vals)
    f2a1.plot([D1grid[0], D1grid[-1]], [D0, D0],
              '--m', label="Sequential fitting")
    f2a1.plot([D1, D1], [D0vals[0], D0vals[-1]], '--m')

    f2a1.plot([D1grid[0], D1grid[-1]], [pg_1step[0], pg_1step[0]],
              '--g', label="Global fitting")
    f2a1.plot([pg_1step[2], pg_1step[2]], [D0vals[0], D0vals[-1]], '--g')
    f2a1.set_xlabel('D1')
    f2a1.set_ylabel('D0')
    print "D0, D1=", D0, D1
    f2a1.set_yscale('log')
    plt.savefig("D1D0_0sigma.png")

# Explore likelihood computation
#    L = 0.0
#    pg = pg_2step
#    plt.close()
#    for i in range(len(data)):
#        fsim = wvtr_vec_global(data[i].ts, data[i].RH, data[i].temp,
#                               data[i].thickness, *pg)
#        inv_sigma2 = 1/(data[i].sigma * data[i].sigma)
#        print "data[",i,"]; sigma=",data[i].sigma
#        print "    err: ", np.sum((fsim - data[i].js)**2)
#        print "    max error: ", np.max((fsim - data[i].js))
#        print "    inv_sigma2 = ", inv_sigma2
#        err = (fsim-data[i].js)/data[i].sigma
#       # plt.plot(err)
#       # plt.show()
#        plt.plot(fsim)
#        plt.plot(data[i].js)
#        plt.show()
#        L = L - 0.5 * np.sum( (fsim - data[i].js)**2 * inv_sigma2)
#
#    print "loglikelihood = ", L
#
# end explore likelihood computation
    # Plot out global and fits of fits
    for i in range(len(data)):
        ax = plt.subplot(nrows, ncols, i)

        thisfit_global_2step = wvtr_vec_global(data[i].ts, data[i].RH,
                                               data[i].temp, data[i].thickness,
                                               *pg_2step)
        thisfit_global = wvtr_vec_global(data[i].ts, data[i].RH, data[i].temp,
                                         data[i].thickness, *pg_1step)
        thisfit = wvtr_vec(data[i].ts, data[i].RH, data[i].temp,
                           data[i].thickness, *pfits[i])

        ax = plt.subplot(nrows, ncols, i)

        l = ax.plot(data[i].ts, data[i].js, 'x')
        ax.plot(data[i].ts, data[i].js+2*data[i].sigma, '--g')
        ax.plot(data[i].ts, data[i].js-2*data[i].sigma, '--g')

        ax.plot(data[i].ts, thisfit_global_2step, '--r', lw=2)
        ax.plot(data[i].ts, thisfit_global, '--m', lw=2)
        ax.plot(data[i].ts, thisfit, '--b', lw=2)
        ax.locator_params(axis='x', nbins=6)
        ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))

        ax.text(0.5, 0.6, data[i].description,
                transform=ax.transAxes, fontsize=10)
        ax.text(0.5, 0.5, data[i].comments,
                transform=ax.transAxes, fontsize=10)
        sc = "{};\n T={}, RH={}".format(data[i].sample_id,
                                        data[i].temp, data[i].RH)
        ax.text(0.5, 0.2, sc, transform=ax.transAxes, fontsize=10)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(10)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flux [kg/m$^2$/s]")

    plt.savefig("fits.png")

    print "MLE solution: ", pg_2step

# Sampling starts here ----------------------------------------------
    do_global = False  # This was `step 1'
    do_global_walkers_prior = True
    do_independent = False
    do_sequential_prior = True
    do_implicit_sampling = False

    p25s = []
    p50s = []
    p75s = []

    # Sampling - for each experiment sequentially, using global model
    ndim, nwalkers = 4, 8

    nSamplesTotal = 1000000
    nSamplesBurnin = 100000
    ensemble_std = 0.05

    if(do_global):
        samples = []
        # Sample all of the experiments together
        pg = pg_2step
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=[data, pg, None, broad_prior])
        pg = np.array(pg)
        if(do_global_walkers_prior):
            pos = get_walkers(nwalkers, broad_prior)
        else:
            pos = [pg + pg
                   * ensemble_std
                   * np.random.randn(ndim) for i in range(nwalkers)]
        plt.close()
        sampler.run_mcmc(pos, nSamplesTotal)
        f, (a1, a2) = plt.subplots(2, 1)
        a1.plot(np.transpose(sampler.chain[:, :, 0]))
        a2.plot(np.transpose(sampler.chain[:, :, 1]))
        plt.savefig("chains_global.png")
        plt.close()

        np.save("samples_global", sampler.chain)
        samples.append(sampler.chain[:, nSamplesBurnin:, :]
                       .reshape((-1, ndim)))

        fig = triangle.corner(samples[0], labels=["D0", "K0", "D1", "K1"])
        plt.savefig("triangle_global.png")
        plt.close()

        p25 = np.percentile(samples[0], [25], axis=0)
        p50 = np.percentile(samples[0], [50], axis=0)
        p75 = np.percentile(samples[0], [75], axis=0)
        p25s.append(p25[0])
        p50s.append(p50[0])
        p75s.append(p75[0])
        print "For global sampling 25th percentile:", p25[0]
        print "For global sampling 50th percentile:", p50[0]
        print "For global sampling 75th percentile:", p75[0]

    if(do_implicit_sampling):

        chain = np.load("samples_global.npy")
        samples = chain[:, nSamplesBurnin:, :].reshape((-1, ndim))
        N = samples.shape[-1]
        M = samples.shape[0]
        pg = np.array(pg_2step)

        scales = np.array(pg)
        plt.close()
        plt.plot(samples)
        plt.show()
        n, bins, patches = plt.hist(samples[:, 0], normed=1,
                                    histtype='bar', rwidth=0.8)
        plt.show()
        f = np.zeros_like(bins)
        i = 0
        for b in bins:
            ppp = pg
            ppp[0] = b
            f[i] = np.exp(lnlike(ppp, data))
            i += 1
        tt = "pg={}".format(pg)
        plt.plot(bins, f)
        plt.title(tt)
        plt.show()

        Hinv = get_H_samples(samples, N, M, scales)

        print "Hinv = ", Hinv
        neff = 1
        evecs, evals = computeHEV(Hinv, neff)

        # This is starting point for IS
        mu = np.matrix(pg/scales).T
        phi = lnlike(pg, data)
        print "mu, phi=", mu, phi

        lb = np.min(samples, axis=0)/scales
        ub = np.max(samples, axis=0)/scales
        tmp = lb[2]
        lb[2] = ub[2]
        ub[2] = tmp
        print "lower bounds:", lb
        print "upper bounds", ub

        NOS = nSamplesTotal
        is_weights, is_samples = do_is(NOS, N, M, evecs, evals, mu, phi,
                                       data, lb, ub, neff, scales, lnlike)
        # print is_weights
        print is_samples
        plt.close()
        print is_samples.shape
        print type(is_samples)
        S = np.squeeze(np.array(is_samples[0, :]))
        print type(S)
        print S.shape

        plt.hist(S)
        ax2 = plt.twinx()
        ax2.plot(bins, f)
        plt.show()

        print 'Effective sample size: ', EffSampleSize(is_weights)
        print 'Quality measure R:', CompR(is_weights)

        CondMean = WeightedMean(is_weights, is_samples)
        print 'Conditional mean: ', CondMean
        print 'Conditional std: ',\
              np.sqrt(WeightedVar(CondMean, is_weights, is_samples))

        rs_map = Resampling(is_weights, is_samples)
        rs_samples = is_samples[:, rs_map]
        S = np.squeeze(np.array(rs_samples[:, :].T))
        fig = triangle.corner(S,
                              labels=["D0", "K0", "D1", "K1"], histcolor='k')
        plt.savefig("is.png")
        plt.show()

        CondMeanRs = WeightedMean(np.ones(NOS)/NOS, rs_samples)
        print 'Conditional mean after resampling: ', CondMeanRs
        print 'Conditional std after resampling: ',\
              np.sqrt(WeightedVar(CondMeanRs, np.ones(NOS)/NOS, rs_samples))

    # Global parameter values for initializing walkers
    pg = pg_2step

    if(do_independent):
        samples = []
        for idata in range(len(data)):
            fname = "_{}.png".format(idata)
            datname = "_{}".format(idata)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=[[data[idata]], pg])
            pg = np.array(pg)
            pos = [pg + pg
                   * ensemble_std
                   * np.random.randn(ndim) for i in range(nwalkers)]
            plt.close()
            sampler.run_mcmc(pos, nSamplesTotal)
            f, (a1, a2) = plt.subplots(2, 1)
            a1.plot(np.transpose(sampler.chain[:, :, 0]))
            a2.plot(np.transpose(sampler.chain[:, :, 1]))
            plt.savefig("chains_indep" + fname)
            plt.close()

            np.save("samples_indep"+datname, sampler.chain)
            samples.append(sampler.chain[:, nSamplesBurnin:, :]
                                  .reshape((-1, ndim)))

            fig = triangle.corner(samples[idata],
                                  labels=["D0", "K0", "D1", "K1"])

            plt.savefig("triangle_indep" + fname)
            plt.close()

            p25 = np.percentile(samples[idata], [25], axis=0)
            p50 = np.percentile(samples[idata], [50], axis=0)
            p75 = np.percentile(samples[idata], [75], axis=0)
            print "For independent sampling, idata:",\
                  idata, " 25th percentile:", p25[0]
            print "For independent sampling, idata:",\
                  idata, " 50th percentile:", p50[0]
            print "For independent sampling, idata:",\
                  idata, " 75th percentile:", p75[0]
            p25s.append(p25[0])
            p50s.append(p50[0])
            p75s.append(p75[0])

        colors = ["m", "g", "b", "y"]
        fig = triangle.corner(samples[0],
                              labels=["D0", "K0", "D1", "K1"],
                              hist_color=colors[0])

        for idata in range(1, len(data)):
            triangle.corner(samples[idata], fig=fig, hist_color=colors[idata])

        plt.savefig("triangle_indep_all.png")
        plt.show()

    if(do_sequential_prior):

        # Build synthetic experiments to do annealing
        # as well as moving through hierarchy
        seq_data = []
        nAnnealingSteps = 4
        sigma_factor = 99.0
        for i in range(nAnnealingSteps):
            seq_data.append([copy.deepcopy(data[0])])
            (seq_data[-1])[0].sigma = data[0].sigma\
                                    * (1.0 + (sigma_factor - 1.0)
                                    * np.exp(-5.0*i/nAnnealingSteps))
        seq_data.append([copy.deepcopy(data[0])])
        seq_data.append(copy.deepcopy(data[0:2]))

        print "Running simulated experiments:"
        for i in range(len(seq_data)):
            print "Step ", i, " experiments: ", len(seq_data[i])
            for ss in seq_data[i]:
                print "    sigma = ", ss.sigma
        # Setup sparse pdf storage to hold posterior
        samples = []
        spdf = sparse_pdf.sparsePdf(4)
        firstExp = True
        for idata in range(len(seq_data)):
            fname = "_{}.png".format(idata)
            datname = "_{}".format(idata)
            if(firstExp):
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                                args=[seq_data[idata], pg,
                                                None, broad_prior])
                # Walkers uniformly distributed throughout bounds
                pos = get_walkers(nwalkers, broad_prior)
                print "First experiment; using broad prior and walkers uniform in bounds"
                print "Walker positions:"
                print pos
                firstExp = False
            else:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                                args=[seq_data[idata], pg, None,
                                                      broad_prior])
                #pos = get_walkers_cluster(nwalkers, broad_prior,
                #                  samples[-1], seq_data[idata-1])
                pos = get_walkers(nwalkers, broad_prior,
                                  samples[-1], seq_data[idata-1])
                print "Subsequent experiment; using prior from binned samples"
                print "and walkers drawn from previous experiment samples"
                print "Walker positions:"
                print pos

            pg = np.array(pg)
            # pos = [pg + pg
            #        * ensemble_std
            #        * np.random.randn(ndim) for i in range(nwalkers)]
            plt.close()
            sampler.run_mcmc(pos, nSamplesTotal)
            f, (a1, a2) = plt.subplots(2, 1)
            a1.plot(np.transpose(sampler.chain[:, :, 0]))
            a2.plot(np.transpose(sampler.chain[:, :, 1]))
            plt.savefig("chains_seq_prior"+fname)
            plt.close()

            np.save("samples_seq_prior"+datname, sampler.chain)
            samples.append(sampler.chain[:, nSamplesBurnin:, :]
                                  .reshape((-1, ndim)))

            print "shape of samples:", samples[idata].shape
            fig = triangle.corner(samples[idata],
                                  labels=["D0", "K0", "D1", "K1"])
            plt.savefig("triangle_seq_prior"+fname)
            plt.close()

            # store the posterior
            spdf.clear()
            lb = np.min(samples[idata], axis=0)
            ub = np.max(samples[idata], axis=0)
            print "lower bounds:", lb
            print "upper bounds:", ub
            spdf.setLowerBounds(lb)
            spdf.setUpperBounds(ub)
            counts = np.array([20, 20, 20, 20], dtype='int')
            spdf.setBinCounts(counts)
            for s in samples[idata]:
                spdf.addSamplePoint(s)
            spdf.normalize()

            # Make a triangle plot from the sparse pdf representation for comparison
            spdf_samples = spdf_weighted_samples(spdf)
            fig = triangle.corner(spdf_samples.transpose(),
                                  labels=["D0", "K0", "D1", "K1"])
            plt.savefig("triangle_seq_prior_spdf"+fname)
            # # Display the jpdf such as it is
            # bins0 = spdf.get_bincens(0)
            # bins1 = spdf.get_bincens(1)
            # bins2 = spdf.get_bincens(2)
            # bins3 = spdf.get_bincens(3)

            # p = np.zeros([len(bins0), len(bins2)])
            # i = 0
            # for b0 in bins0:
            #     j = 0
            #     for b2 in bins2:
            #         p[i, j] = spdf.getProbability([b0, bins1[0], b2, bins3[0]])
            #         j += 1
            #     i += 1
            # print "The constructed distribution:"
            # print p

            p25 = np.percentile(samples[idata], [25], axis=0)
            p50 = np.percentile(samples[idata], [50], axis=0)
            p75 = np.percentile(samples[idata], [75], axis=0)
            print "For sequential sampling, previous posterior as prior, exp:",\
                  idata, " 25th percentile:", p25[0]
            print "For sequential sampling, previous posterior as prior, exp:",\
                  idata, " 50th percentile:", p50[0]
            print "For sequential sampling, previous posterior as prior, exp:",\
                  idata, " 75th percentile:", p75[0]
            p25s.append(p25[0])
            p50s.append(p50[0])
            p75s.append(p75[0])

        colors = ["m", "g", "b", "y"]
        fig = triangle.corner(samples[0],
                              labels=["D0", "K0", "D1", "K1"],
                              hist_color=colors[0])

        for idata in range(1, len(data)):
            triangle.corner(samples[idata], fig=fig, hist_color=colors[idata])

        plt.savefig("triangle_seq_prior_all.png")
        plt.show()
    np.save("p25s", np.array(p25s))
    np.save("p50s", np.array(p50s))
    np.save("p75s", np.array(p75s))
