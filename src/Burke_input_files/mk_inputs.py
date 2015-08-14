from xlrd import open_workbook
import xlrd
import numpy as np
import matplotlib.pyplot as plt


def read_expts(f, ra, rb, sheet):
    wb = open_workbook(f)
    s = wb.sheets()[sheet]
    hdr = s.row_slice(1)[1:39]
    print hdr
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
        pmfname = "premix." + fd['name']
        pmfile = open(pmfname,'w')
        pmfile.write('FREE\n')
        pmfile.write('ENRG\n')
        pmfile.write('MULT\n')
        pmfile.write('FLRT ' + "{}".format(fd['flrt']) + "\n")
        if int(fd['psteps']) > 0:
            pmfile.write('PRES ' + "{}".format(fd['pstart']) + "\n")
        else:
            pmfile.write('PRES ' + "{}".format(fd['P']) + "\n")
        pmfile.write('NPTS ' + "{0:0d}".format(int(fd['npts'])) + "\n")
        pmfile.write('XEND ' + "{}".format(fd['xend']) + "\n")
        pmfile.write('XCEN ' + "{}".format(fd['xcen']) + "\n")
        pmfile.write('TFIX ' + "{}".format(fd['tfix']) + "\n")
        pmfile.write('WMIX ' + "{}".format(fd['wmix']) + "\n")
        pmfile.write('MOLE\n')
        pmfile.write('REAC H2 ' + "{}".format(fd['H2']) + "\n")
        pmfile.write('REAC O2 ' + "{}".format(fd['O2']) + "\n")
        pmfile.write('REAC HE ' + "{}".format(fd['He']) + "\n")
        if 'Ar' in fd:
            pmfile.write('REAC AR ' + "{}".format(fd['Ar']) + "\n")
            xAr = fd['Ar']
        else:
            xAr = 0.0
        xN2 = 1.0 - fd['H2'] - fd['O2'] - fd['He'] - xAr
        pmfile.write('REAC N2 ' + "{}".format(xN2) + "\n")
    
        xH2O_prod = min(fd['H2'], 2*fd['O2'])
        xH2_prod = max(0.0, fd['H2'] - xH2O_prod)
        xO2_prod = max(0.0, fd['O2'] - 0.5*xH2O_prod)
    
        tot = xH2O_prod+xO2_prod+xH2_prod+xAr+fd['He']+xN2
        xH2O_prod = xH2O_prod / tot
        xH2_prod = xH2_prod / tot
        xO2_prod = xO2_prod / tot
        xAr_prod = xAr / tot
        xHe_prod = fd['He'] / tot
        xN2 = xN2 / tot
    
        if 'Ar' in fd:
            pmfile.write('PROD AR ' + "{}".format(xAr_prod) + "\n")
        pmfile.write('PROD HE ' + "{}".format(xHe_prod) + "\n")
        pmfile.write('PROD N2 ' + "{}".format(xN2) + "\n")
        pmfile.write('PROD H2 ' + "{}".format(xH2_prod) + "\n")
        pmfile.write('PROD O2 ' + "{}".format(xO2_prod) + "\n")
        pmfile.write('PROD H2O ' + "{}".format(xH2O_prod) + "\n")
    
        pmfile.write('INTM   HO2     0.0001\n')
        pmfile.write('INTM   O       0.0001\n')
        pmfile.write('INTM   H2O2    0.0001\n')
        pmfile.write('INTM   H       0.01\n')
        pmfile.write('INTM   OH      0.01\n')
    
        pmfile.write('ATOL ' + "{}".format(fd['atol']) + "\n")
        pmfile.write('RTOL ' + "{}".format(fd['rtol']) + "\n")
        pmfile.write('ATIM ' + "{}".format(fd['atim']) + "\n")
        pmfile.write('RTIM ' + "{}".format(fd['rtim']) + "\n")
    
        pmfile.write('PRNT  1\n')
    
        pmfile.write('TIME ' + "{:0d} {}".format(int(fd['ntime']),fd['dt_time']) + "\n")
        pmfile.write('TIM2 ' + "{:0d} {}".format(int(fd['ntim2']),fd['dt_tim2']) + "\n")
    
    # estimated temperature profile
        Tx, Tv = temp_profile(0.0,fd['xend'],fd['xcen'],fd['Twidth'], 295.0,fd['T'])
        for (x,v) in zip(Tx,Tv):
            pmfile.write("TEMP {}  {}\n".format(x,v))
    
        pmfile.write('CNTN\n')
        pmfile.write('END\n')
        pmfile.write('GRAD  0.9\n')
        pmfile.write('CURV  0.9\n')
        if fd['psteps'] > 0:
            ps = annealing_schedule_exp_up(fd['pstart'], fd['P'], int(fd['psteps'])+1, fd['k3'])
            #ps = np.linspace(fd['pstart'], fd['P'],int(fd['psteps'])+1)
            for pp in ps[1:]:
                pmfile.write('CNTN\n')
                pmfile.write('END\n')
                pmfile.write("PRES {}\n".format(pp))
                pmfile.write('CNTN\n')
                pmfile.write('END\n')
                pmfile.write('GRAD  0.9\n')
                pmfile.write('CURV  0.9\n')
        else:
            pmfile.write('END\n')

        if fd['refinesteps'] > 0:
            gs = annealing_schedule_lin(0.9,fd['finalGRAD'],int(fd['refinesteps']))
            cs = annealing_schedule_lin(0.9,fd['finalCURV'],int(fd['refinesteps']))
            for g,c in zip(gs,cs):
                pmfile.write('CNTN\n')
                pmfile.write('END\n')
                pmfile.write('GRAD  {}\n'.format(g))
                pmfile.write('CURV  {}\n'.format(c))
            pmfile.write('END\n')
        else:
            pmfile.write('END\n')


                
            
    
        pmfile.close()
        
    
        print fd['name'] + ".type = PREMIXReactor"
        print fd['name'] + ".data = " + "{}".format(fd['su'])
        print fd['name'] + ".premix_input_path = ./Burke_input_files/"
        print fd['name'] + ".premix_input_file = premix." +fd['name']
        print fd['name'] + ".measurement_error = " + "{}".format(fd['sigma_s'])
        print fd['name'] + ".num_sol_pts = 1000"
        print "\n\n"


flmdat1 = read_expts('Burke_data.xlsx',3,48,0)
flmdat2= read_expts('Burke_data.xlsx',3,35,1)

write_premix_files(flmdat1)
write_premix_files(flmdat2)


