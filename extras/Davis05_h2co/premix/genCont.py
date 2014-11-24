H2st = 0.05
H2nd = 0.029
nSteps = 11

CO = 0.141
O2 = 0.184

for i in range(nSteps):
    eta = float(i)/(nSteps-1)
    H2 = H2st*(1-eta) + H2nd*eta
    N2 = 1 - (H2 + CO + O2)
    print '/'
    print 'REAC H2',H2
    print 'REAC O2',O2
    print 'REAC CO',CO
    print 'REAC N2',N2
    print 'CNTN'
    print 'END'
    

    
