import os
import uuid

def get_pbsid():
    id = os.popen("echo $PBS_JOBID").read().strip()
    if id == "" or id.isspace():
        id = str(uuid.uuid4())
    return id

def get_reflog():
    rl = os.popen("git rev-parse HEAD").read()
    return rl

def get_patch():
    patch = os.popen("git show").read()
    return patch



if __name__ == "__main__":

    id = get_pbsid()
    print id
    rl = get_reflog()
    patch = get_patch()

    f = open('test.log', 'a')
    f.write(id + "  " + rl)
    f.close()
    
    f = open(id + ".src", 'w')
    f.write(patch)
    f.close()
    

     

    
