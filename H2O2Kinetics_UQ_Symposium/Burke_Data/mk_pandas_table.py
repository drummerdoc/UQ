import sys
import numpy as np
import pandas as pd
f = open("BurkeCDJ10_data.txt", "r")
data = np.array(f.read().split())
f.close()
i = 1
d = data.reshape(-1,14)
#print d

cols = ["phi", "Tf", "p", "XH2", "XCO", "XO2", "XHE", "XAR", "XCO2", "Rl", "Ru", "Su", "f", "sigmaf"]
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
cols = ["phi", "Tf", "p", "XH2"]
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



#for d in data[::14]:
#
#    sys.stdout.write(d.rstrip())
#    sys.stdout.write(" ")
#    if( i%14 == 0 ):
#        sys.stdout.write("\n")
#    i +=1
#    
#df = df1.combine_first(df2).fillna(value=0)
#df = pd.concat([df1,df2],axis=1).fillna(value=0)
#print df
df1['source'] = 'CNF'
df1['col'] = 'g'
df2['col'] = 'm'
df = pd.concat([df1,df2],join='outer',axis=0).fillna(value=0)
df5 = df.loc[(df['Tf']==1600) & (df['p']==5) & (df['XCH4']==0) & (df['XCO']==0.0) & (df['XCO2']==0.0) & (df['XAR']==0)]

