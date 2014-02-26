#ifndef MINPACKstruct_H
#define MINPACKstruct_H

#include <iostream>

#include <cminpack.h>
#include <lapacke.h>

// A simple struct to hold all the parameters, experiment data and some work space
// to be accessible to the MINPACK function
struct MINPACKstruct
{
  MINPACKstruct(ChemDriver& cd, Real _param_eps)
    : parameter_manager(cd), expt_manager(parameter_manager,cd),
    param_eps(_param_eps), num_work_arrays(4), work_array_len(-1) {}

  struct LAPACKstruct
  {
    lapack_int DGESVD_wrap() {
      lapack_int info;
      if (lwork == -1) {
        double work_query;
        info = LAPACKE_dgesvd_work(matrix_order,jobu,jobvt,m,n,a.dataPtr(),lda,s.dataPtr(),
                                   u.dataPtr(),ldu,vt.dataPtr(),ldvt,&work_query,lwork);
        lwork = (lapack_int)work_query;
        work.resize(lwork);
      }
      else {
        /* Call middle-level interface */
        info = LAPACKE_dgesvd_work(matrix_order,jobu,jobvt,m,n,a.dataPtr(),lda,s.dataPtr(),
                                   u.dataPtr(),ldu,vt.dataPtr(),ldvt,work.dataPtr(),lwork);
      }
      BL_ASSERT(info == 0);
      return info;
    }

    lapack_int DSYEV_wrap() {
      lapack_int info;
      if (lwork == -1) {
        double work_query;
        info = LAPACKE_dsyev_work(matrix_order,jobz,uplo,n,a.dataPtr(),lda,s.dataPtr(),
                                  &work_query,lwork);
        lwork = (lapack_int)work_query;
        work.resize(lwork);
      }
      else {
        /* Call middle-level interface */
        info = LAPACKE_dsyev_work(matrix_order,jobz,uplo,n,a.dataPtr(),lda,s.dataPtr(),
                                  work.dataPtr(),lwork);
      }
      BL_ASSERT(info == 0);
      return info;
    }

    void ResizeWork(int work_array_len) {
      matrix_order = LAPACK_ROW_MAJOR; // Order of matrices

      jobu  = 'a';             // 'a' -> all M columns of U are returned
      jobvt = 'a';             // 'a' -> all N rows of V**T are returned
      jobz  = 'v';             // 'V' -> return vectors and values 'N' otherwise
      uplo  = 'u';             // 'U' -> A is upper-triangular
      m     = work_array_len;  // Number of rows of matrix
      n     = work_array_len;  // Number of columns of matrix
      a.resize(n*m);           // The matrix A
      lda   = n;               // Leading dimension of matrix a
      s.resize(std::min(m,n)); // Singular values of a
      u.resize(m*m);           // The matrix U
      ldu   = m;               // Leading dimension of matrix u
      vt.resize(n*n);          // The matrix VT
      ldvt  = n;               // Leading dimension of matrix vt

      // Query for work size
      lwork = -1;              // size of work, -1 for query
      //lapack_int info = DGESVD_wrap();
      lapack_int info = DSYEV_wrap();
      BL_ASSERT(info == 0);
    }
    char jobu,jobvt,jobz,uplo;
    lapack_int m, n, lda, ldu, ldvt, lwork;
    Array<double> a, s, u, vt, work;
    int matrix_order;
  };

  void ResizeWork() {
    if (work_array_len != parameter_manager.NumParams()) {
      work_array_len = parameter_manager.NumParams();
      work.resize(num_work_arrays);
      for (int i=0; i<num_work_arrays; ++i) {
        work[i].resize(work_array_len);
      }
      lapack_struct.ResizeWork(work_array_len);
    }
  }

  Array<Real>& Work(int i) {
    BL_ASSERT(i<num_work_arrays);
    return work[i];
  }
  ParameterManager parameter_manager;
  ExperimentManager expt_manager;
  LAPACKstruct lapack_struct;
  Array<Array<Real> > work;
  Real param_eps;
  int num_work_arrays, work_array_len;
};

#endif
