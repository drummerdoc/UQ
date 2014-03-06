#ifndef PREMIXSOL_H
#define PREMIXSOL_H

// This is basic container to store a full solution from premix
// that can be used for restarts (e.g., more data than strictly
// necessary to extract an 'observation').
//
// The raw solution vector, plus some basics about the number of species
// and so on to prevent blunt errors. 
//

struct PremixSol
{
    PremixSol( int nc, int sz ){
        ncomp = nc;
        maxgp = sz;
        nextra = 10; // 'extra' values - p, flowrate, flame speed, etc.
        solvec = new double[ncomp*maxgp + nextra];
        ngp = -1;

    }
    ~PremixSol(){
        delete solvec;
    }

    int ncomp;
    int maxgp;
    int nextra;
    int ngp;
    
    // This is a rather dumb object
    // Needs capacity of 
    // (maximum number of grid points)*(solution size)
    // where solution size includes the grid.
    // Inside premix copy
    // savesol[j] = X[j] where j=0,gridsize < jmax
    // savesol[j + (n*jmax)] where n = 1,natj; natj<maxstate)
    // Stores the premix solution to restart from
    double * solvec;

    PremixSol & operator=(const PremixSol &a) {                                                                                                                                           

        if( &a==this ){
            return *this;
        }
        ncomp = a.ncomp;
        maxgp = a.maxgp;
        nextra = a.nextra;
        ngp = a.ngp;
        if( solvec != NULL ){
            delete solvec;
        }
        int sz = ncomp*maxgp + nextra;
        solvec = new double[sz];
        for(int i=0; i<sz; i++){
            solvec[i] = a.solvec[i];
        }
        
        return *this;
    }


};

#endif
