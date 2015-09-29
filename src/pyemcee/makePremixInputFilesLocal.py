import string
import sys

f = open(sys.argv[1])
lines = f.readlines()
f.close

INPUT_PATH="./PREMIX_INPUTS/"
BASELINE_PATH="./PREMIX_SOLN"

for l in lines:
    tokens = string.split(l.strip(),'=')
    if (len(tokens)==2):
        ptokens = string.split(tokens[0].strip(),".")
        if (len(ptokens)>1):
            param = ptokens[-1]
            if (param == "premix_input_file"):
                vtokens = string.split(tokens[1].strip(),"/")
                input_file = vtokens[-1]
                print tokens[0] + "= " + input_file
            elif (param == "premix_input_path"):
                input_path = INPUT_PATH
                print tokens[0] + "= " + input_path
            elif (param == "baseline_soln_file"):
                vtokens = string.split(tokens[1].strip(),"/")
                print tokens[0] + "= " + BASELINE_PATH+"/"+vtokens[-1]
            else:
                print l.strip()
        else:
            print l.strip()
    else:
        print l.strip()
