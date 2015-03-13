#ifndef SPARSE_PDF_H
#define SPARSE_PDF_H

void test();
#include <vector.h>
#include <map>


struct sparsePdf {
    sparsePdf(int ndim);
    ~sparsePdf(){}
    void setLowerBounds( std::vector<double> bot);
    void setUpperBounds( std::vector<double> top);
    void setBinCounts( std::vector<int> counts);
    void clear();
    void normalize();

    std::vector<double> get_bincens(int dim);
    void addSamplePoint( std::vector<double> pt); // make one also for a list of vectors
    void incrementBin( std::vector<int> idx);
    double getProbability( std::vector<double> pt);    
    //
    
    std::map<std::vector<int>, double> jpdf;
    bool _validUpperBounds;
    bool _validLowerBounds;
    bool _ready_to_add;
    std::vector<double> bin_minval;
    std::vector<double> bin_maxval;
    std::vector<int> bin_counts;
    std::vector<double> bin_delta;
    int _ndim;
};

#endif

