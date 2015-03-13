#include "sparse_pdf.H"

#include<stdio.h>
#include<iostream>

void test(){
    printf("Testing...\n");
    return;
}

sparsePdf::sparsePdf(int ndim){
    _ndim = ndim;
    bin_minval.resize(_ndim);
    bin_maxval.resize(_ndim);
    printf("Testing from sparsePdf object\n");
    _validLowerBounds = false;
    _validUpperBounds = false;
    _ready_to_add = false;
}

void sparsePdf::setLowerBounds( std::vector<double> vals ){
    if( vals.size() != _ndim ){
        std::cerr << "Wrong length of value vector for lower bounds; _ndim = " << _ndim << std::endl;
    }
    bin_minval = vals;
    _validLowerBounds = true;
}

void sparsePdf::setUpperBounds( std::vector<double> vals ){
    if( vals.size() != _ndim ){
        std::cerr << "Wrong length of value vector for upper bounds; _ndim = " << _ndim << std::endl;
    }
    bin_maxval = vals;
    _validUpperBounds = true;
}

void sparsePdf::setBinCounts( std::vector<int> counts ){
    if( !( _validUpperBounds && _validLowerBounds) ) {
        std::cerr << "Bounds not properly initialized" << std::endl;
        return;
    }
    if( counts.size() != _ndim ){
        std::cerr << "Wrong length of value vector; _ndim = " << _ndim << std::endl;
    }
    bin_counts = counts;
    bin_delta.resize(bin_counts.size());
    for(int i=0; i<_ndim; ++i) {
        bin_delta[i] = (bin_maxval[i] - bin_minval[i])/bin_counts[i];

    }
    _ready_to_add = true;
}

void sparsePdf::clear() {
    jpdf.clear();
    _ready_to_add = true;

}

void sparsePdf::normalize() {
    double sumval = 0.0;
    for (std::map<std::vector<int>,double>::iterator ii=jpdf.begin(); ii!=jpdf.end(); ++ii){
        sumval += (*ii).second;
    }
    if( sumval < 0 ){
        std::cerr << "jpdf empty!\n";
        return;
    }
    for (std::map<std::vector<int>,double>::iterator ii=jpdf.begin(); ii!=jpdf.end(); ++ii){
        (*ii).second = (*ii).second/sumval;;
    }
    _ready_to_add = false;

}

std::vector<double> sparsePdf::get_bincens(int dim) {
    std::vector<double> bincens;

    bincens.resize(bin_counts[dim]);
    for(int i=0; i< bin_counts[dim]; ++i){
        bincens[i] = bin_minval[dim] + i*bin_delta[dim] + 0.5*bin_delta[dim];
    }
    return bincens;

}

void sparsePdf::addSamplePoint( std::vector<double> pt) {
    std::vector<int> idx;
    if( pt.size() != _ndim ){
        std::cerr << "Wrong number of dimensions for pt; " << _ndim << std::endl;
        return;
    }
    if( !_ready_to_add ) {
        std::cerr << "Can not add to jpdf after normalization!" << std::endl;
        return;
    }
    idx.resize(_ndim);
    for(int i=0; i<_ndim; ++i){
        idx[i] = (pt[i] - bin_minval[i] )/bin_delta[i];
    }
    incrementBin( idx );

}

void sparsePdf::incrementBin( std::vector<int> idx ){
    if( jpdf.find(idx) == jpdf.end() ){
        jpdf[idx] = 1.0;
    }
    else{
        jpdf[idx] = jpdf[idx] + 1.0;
    }
}

double sparsePdf::getProbability( std::vector<double> pt ) {
    std::vector<int> idx;
    if( pt.size() != _ndim ){
        std::cerr << "Wrong number of dimensions for pt; " << _ndim << std::endl;
        return -1;
    }
    idx.resize(_ndim);
    for(int i=0; i<_ndim; ++i){
        idx[i] = (pt[i] - bin_minval[i] )/bin_delta[i];
    }
    if( jpdf.find(idx) == jpdf.end() ){
        return 0.0;
    }
    else{
        return jpdf[idx];
    }


}

