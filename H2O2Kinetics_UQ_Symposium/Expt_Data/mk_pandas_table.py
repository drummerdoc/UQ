import sys
import numpy as np
import pandas as pd
f = open("BurkeCDJ10_data.txt", "r")
data = np.array(f.read().split())
f.close()
i = 1
d = data.reshape(-1,14)
#print d

cols = ["phi", "Tf", "p0", "XH2", "XCO", "XO2", "XHE", "XAR", "XCO2", "Rl", "Ru", "Su", "f", "sigmaf"]
dd = {}
for i in range(14):
    s = pd.Series(d[:,i].copy()).astype(np.float32)
    dd[cols[i]] = s

df1 = pd.DataFrame(dd)

#print df


dd = {}
dl = []
f = open("BurkeDJ11_data.txt","r")
data = np.array(f.readline().split())
d = data.reshape(-1,4)
cols = ["phi", "Tf", "p0", "XH2"]
for i in range(4):
    s = pd.Series(d[:,i].copy()).astype(np.float32)
    dd[cols[i]] = s

data = np.array(f.readline().split())
d = data.reshape(-1,2)
cols = ["XCH4", "XO2"]
for i in range(2):
    s = pd.Series(d[:,i].copy()).astype(np.float32)
    dd[cols[i]] = s
    dl.append(s)
  
 

data = np.array(f.readline().split())
n = d.shape[0]
d = np.zeros([n,3])
d[:,0:2] = data[:2*n].reshape(-1,2)
d[:,2] = data[2*n:3*n]

print "line with XHe=", d[:,0]
print d[:,1]
print d[:,2]
cols = ["XHE", "Rl", "Ru"]
for i in range(3):
    s = pd.Series(d[:,i].copy()).astype(np.float32)
    dd[cols[i]] = s


data = np.array(f.readline().split())
d = data.reshape(-1,2)
cols = ["Su", "f"]
for i in range(2):
    s = pd.Series(d[:,i].copy()).astype(np.float32)
    dd[cols[i]] = s
       
data = np.array(f.readline().split())
d = data.reshape(-1,1)
cols = ["sigmaf"]
for i in range(1):
    s = pd.Series(d[:,i].copy()).astype(np.float32)
    dd[cols[i]] = s

df2 = pd.DataFrame(dd)



df1['Ref'] = 'BurkeCDJ10'
df2['Ref'] = 'BurkeDJ11'
df1.drop('Rl',axis=1,inplace=True)
df2.drop('Rl',axis=1,inplace=True)
df1.drop('Ru',axis=1,inplace=True)
df2.drop('Ru',axis=1,inplace=True)
df1['T0'] = 295
df2['T0'] = 295
df2['sigma'] = df2['sigmaf']*df2['Su']/df2['f']
df1['sigma'] = df1['sigmaf']*df1['Su']/df1['f']
df1.drop('sigmaf',axis=1,inplace=True)
df2.drop('sigmaf',axis=1,inplace=True)
df1.drop('f',axis=1,inplace=True)
df2.drop('f',axis=1,inplace=True)

df1['ID']=''
for i in df1.index:
    df1['ID'][i] = df1['Ref'][i]+"_{}".format(i)
df2['ID']=''
for i in df2.index:
    df2['ID'][i] = df2['Ref'][i]+"_{}".format(i)
df1 = df1.set_index('ID')
df2 = df2.set_index('ID')

df = pd.concat([df1,df2],join='outer',axis=0).fillna(value=0)
df.rename(columns={'Su':'Target'},inplace=True)
df5 = df.loc[(df['Tf']==1600) & (df['p0']==5) & (df['XCH4']==0) & (df['XCO']==0.0) & (df['XCO2']==0.0) & (df['XAR']==0)]

# Get Davis data
d = {}
with open ("Davis_expts.txt",'r') as f:
    hdr = f.readline().rstrip().split()
    print "hdr: ", hdr
    for i in range(len(hdr)):
        d[hdr[i]] = []
    for l in f:
        l = l.rstrip().split()
        print len(hdr), len(l), l
        for i in range(len(hdr)):
            if( hdr[i] == 'ID' or hdr[i] == 'Bath' or hdr[i] == 'Ref' ):
                d[hdr[i]].append(l[i])
            else:
                d[hdr[i]].append(np.float32(l[i]))
        
davis = pd.DataFrame(d).set_index('ID')

# Grab experiments with no C, P>=1 from Burke
burke_set = df.loc[(df['p0']>=1.0) & (df['XCH4']==0) & (df['XCO']==0.0) & (df['XCO2']==0.0) &
        (df['phi']>0.3) & (df['phi']<=1.0) & (df['Tf']<1700)].copy()
burke_set.drop('XCH4',axis=1, inplace=True)
burke_set.drop('XCO',axis=1, inplace=True)
burke_set.drop('XCO2',axis=1, inplace=True)

davis_no_co = davis.loc[(davis['XCO']==0)]
davis_no_co['phi_new'] = davis_no_co['XH2']/davis_no_co['XO2']/2.0
davis['phi'].fillna(value=davis_no_co['phi_new'], inplace=True)

davis_set = davis.loc[(davis['XCO']<=0.000001) & (davis['phi']<=1)].copy()

expts = pd.concat([burke_set,davis_set],join='outer',axis=0)

expts.XAR = expts.XAR.fillna(value=0)
#expts.XCO2 = expts.XCO2.fillna(value=0)
expts.XCO = expts.XCO.fillna(value=0)
#expts.XCH4 = expts.XCH4.fillna(value=0)
expts.XH2O = expts.XH2O.fillna(value=0)
expts.XHE = expts.XHE.fillna(value=0)

expts.drop('XAR',axis=1,inplace=True)
expts.drop('XCO',axis=1,inplace=True)
expts.drop('XH2O',axis=1,inplace=True)

a = expts.to_latex(float_format=lambda x: '%.1f' % x)
f = open('expts.tex','w')
f.write(a.encode("UTF-8"))
f.close()
