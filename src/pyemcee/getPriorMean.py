import string
import sys

f = open(sys.argv[1])
lines = f.readlines()
f.close

prior_means = {}
active_parameters = []
for l in lines:
    tokens = string.split(l.strip(),'=')
    if (len(tokens)==2):
        ptokens = string.split(tokens[0].strip(),".")
        if (len(ptokens)>1):
            myP = ptokens[0].strip()
            param = ptokens[-1]
            if (param == "prior_mean"):
                prior_means[myP] = tokens[1].strip()
        elif (len(ptokens)==1):
            myP = ptokens[0].strip()
            if (myP == "parameters"):
                active_parameters = string.split(tokens[1].strip()," ")
            
active_parameter_values = [prior_means[ap] for ap in active_parameters]
print " ".join(active_parameter_values),"0" # Fake an F at the end so it looks like a JBB format
