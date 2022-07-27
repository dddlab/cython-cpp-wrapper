#include "wrap.h"
#include "core.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>


using namespace std;
using namespace Eigen;

void gconcord(double* s, int sCol, int method,
              double* lam1, double lam2, 
              double epstol, int maxitr, int steptype, 
              double* out, int* outi, int* outj,
              double* update_hist, int* iter_count)
{
    
    MatrixXd S = Map<Matrix<double, Dynamic, Dynamic> > (s, sCol, sCol);
    MatrixXd LambdaMat = Map<Matrix<double, Dynamic, Dynamic> > (lam1, sCol, sCol);
    MatrixXd Out = Map<Matrix<double, Dynamic, Dynamic> > (out, sCol, sCol);
    Map<VectorXd> Update_hist(update_hist, maxitr);
    Map<VectorXi> Iter_count(iter_count, 1);
    SparseMatrix<double> X(sCol, sCol);
    X = Out.sparseView();

    // Update_hist[0,0] = 123.0;
    // Iter_count[0] = 5;
    
    // Update_hist(0,0) = 123.0;
    // Iter_count(0) = 5;

    if( method == 1 ){
        ccorig(S, Out, LambdaMat, lam2, epstol, maxitr);
        X = Out.sparseView();
    }else if( method == 2 ){
        // ccista(S, X, LambdaMat, lam2, epstol, maxitr, steptype);
        ccista(S, X, LambdaMat, lam2, epstol, maxitr, steptype, Update_hist, Iter_count);
    }else if( method == 3 ){
        ccfista(S, X, LambdaMat, lam2, epstol, maxitr, steptype);
    }
    
    int j = 0;
    for(int k=0; k < X.outerSize(); ++k){
        for(SparseMatrix<double>::InnerIterator it(X, k); it; ++it){
            out[j] = it.value();
            outi[j] = it.row();
            outj[j] = it.col();
            j++;
        }
    }
    
}
