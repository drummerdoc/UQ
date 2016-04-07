from xlrd import open_workbook
import xlrd
import numpy as np
import matplotlib.pyplot as plt


def read_expts(f, ra, rb, sheet):
    wb = open_workbook(f)
    s = wb.sheets()[sheet]
    hdr = s.row_slice(1)[1:39]
    flmdata = []
    for row in range(ra,rb):
        rdata = s.row_slice(row)[1:39]
        thisflm = {}
        for h, cell in zip(hdr,rdata):
            thisflm[str(h.value)] = cell.value
        flmdata.append(thisflm)
    return flmdata

def annealing_schedule_exp_up(p0, p1, n, k3):
    k2 = (p1 - p0)/(1.0 - np.exp(-k3*n*1.0))
    k1 = k2 + p0
    p = np.zeros([n+1])
    #print k1, k2, p0, p1, k3
    for nn in range(0,n+1):
        p[nn] = k1 - k2*np.exp(-k3*nn*1.0)
    return p[1:]

def annealing_schedule_lin(p0, p1, n):
    b = p0
    m = (p1-p0)/n
    p = np.zeros([n+1])
    for nn in range(0,n+1):
        p[nn] = b + m*nn
    return p[1:]

def temp_profile(x1, x2, xcen, w, T1, T2):
    xgrid = np.linspace(x1,x2,20)
    xu_cen = (xcen - x1)*2.0/(x2-x1) - 1.0
    w_u = 2.0*w/(x2-x1)
    xu_grid = (2.0*(xgrid - x1)/(x2-x1) - 1.0 - xu_cen )*2.0*np.exp(1)/w_u
    T = 0.5*(np.tanh( (xu_grid))+1)*(T2-T1) + T1
    return xgrid, T

def write_premix_files(flmdat):
    for fd in flmdat:
        pmfname = "ct_" + fd['name'] + ".py"
        pmfile = open(pmfname,'w')
        pmfile.write('import cantera as ct\n')
        pmfile.write('import numpy as np\n')
        if int(fd['psteps']) > 0:
            pmfile.write('p = ct.one_atm * ' + "{}".format(fd['pstart']) + "\n")
        else:
            pmfile.write('p = ct.one_atm * ' + "{}".format(fd['P']) + "\n")

        pmfile.write('Tin = 295\n')

        if 'Ar' in fd:
            pmfile.write('XAR = ' + "{}".format(fd['Ar']) + "\n")
            xAr = fd['Ar']
        else:
            xAr = 0.0
        xN2 = 1.0 - fd['H2'] - fd['O2'] - fd['He'] - xAr
        pmfile.write('XN2 = ' + "{}".format(xN2) + "\n")
        pmfile.write('reactants = \'%s:%g, %s:%g, %s:%g, %s:%g, %s:%g\'\n'
            % ('H2',fd['H2'],'O2',fd['O2'],'HE',fd['He'],'AR',xAr,'N2',xN2))

        pmfile.write('initial_grid = np.linspace(0.0,%g,%d)\n' % (fd['xend'],int(fd['npts'])))
        #pmfile.write('tol_ss = [%g, %g]\n' % (fd['atol'],fd['rtol']))
        #pmfile.write('tol_ts = [%g, %g]\n' % (fd['atol'],fd['rtol']))
        pmfile.write('tol_ss = [1.e-5, 1.e-13]\n')
        pmfile.write('tol_ts = [1.e-4, 1.e-13]\n')
        pmfile.write('loglevel = 0\n')
        pmfile.write('refine_grid = True\n')
        pmfile.write('ct.add_directory(\'/home/marc/src/CCSE/Combustion/Chemistry/data/BurkeDryer_mod\')\n')
        pmfile.write('gas = ct.Solution(\'BurkeDryer_mod.cti\', \'gas\')\n')
        pmfile.write('gas.TPX = Tin, p, reactants\n')
        pmfile.write('f = ct.FreeFlame(gas, initial_grid)\n')
        pmfile.write('f.flame.set_steady_tolerances(default=tol_ss)\n')
        pmfile.write('f.flame.set_transient_tolerances(default=tol_ts)\n')
        pmfile.write('f.energy_enabled = False\n')
        pmfile.write('f.transport_model = \'Mix\'\n')
        pmfile.write('f.set_max_jac_age(10, 10)\n')
        pmfile.write('f.set_time_step(1e-5, [2, 5, 10, 20])\n')
        pmfile.write('f.set_time_step(1e-7, [20, 50, 100, 200])\n')
        pmfile.write('f.solve(loglevel=loglevel, refine_grid=False)\n')
        pmfile.write('f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)\n')
        pmfile.write('f.set_refine_criteria(ratio=3, slope=0.02, curve=0.04)\n')
        pmfile.write('f.energy_enabled = True\n')

        pmfile.write('f.solve(loglevel=loglevel, refine_grid=refine_grid)\n')
        if int(fd['psteps']) > 0:
            pmfile.write('f.P = ct.one_atm * %g\n' % (fd['P']))
            pmfile.write('f.solve(loglevel=loglevel, refine_grid=False)\n')
            pmfile.write('f.solve(loglevel=loglevel, refine_grid=refine_grid)\n')

        pmfile.write('print(\'mixture-averaged flamespeed = {0:7f} m/s\'.format(f.u[0]))\n')
        pmfile.close()

        # print fd['name'] + ".type = PREMIXReactor"
        # print fd['name'] + ".data = " + "{}".format(fd['su'])
        # print fd['name'] + ".premix_input_path = ./Burke_input_files/"
        # print fd['name'] + ".premix_input_file = premix." +fd['name']
        # print fd['name'] + ".measurement_error = " + "{}".format(fd['sigma_s'])
        # print fd['name'] + ".num_sol_pts = 1000"
        # print "\n\n"


flmdat1 = read_expts('Burke_data.xlsx',3,48,0)
flmdat2= read_expts('Burke_data.xlsx',3,35,1)

write_premix_files(flmdat1)
write_premix_files(flmdat2)


