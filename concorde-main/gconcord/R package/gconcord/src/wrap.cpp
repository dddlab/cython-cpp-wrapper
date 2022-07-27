#include <RcppEigen.h>
#include "core.h"

using namespace Rcpp;
using namespace std;


// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
SEXP ccista_conv(
    const Eigen::MatrixXd & S,
    const Eigen::MappedSparseMatrix<double> & X0_,
    const Eigen::MatrixXd & LambdaMat,
    double lambda2,
    double epstol,
    int maxitr,
    int steptype
  )
{
  Eigen::SparseMatrix<double> X(X0_);
  //Eigen::VectorXd maxd(maxitr);

  ccista(S, X, LambdaMat, lambda2, epstol,
         maxitr, steptype);

  // return List::create(Named("out") = wrap(X), Named("maxdiff") = wrap(maxd), Named("nitr") = nitr);
  return List::create(Named("out") = wrap(X));
}

// [[Rcpp::export]]
SEXP ccfista_conv(
    const Eigen::MatrixXd & S,
    const Eigen::MappedSparseMatrix<double> & X0_,
    const Eigen::MatrixXd & LambdaMat,
    double lambda2,
    double epstol,
    int maxitr,
    int steptype
)
{
  Eigen::SparseMatrix<double> X(X0_);
  //Eigen::VectorXd maxd(maxitr);

  ccfista(S, X, LambdaMat, lambda2, epstol,
          maxitr, steptype);

  //return List::create(Named("out") = wrap(X), Named("maxdiff") = wrap(maxd), Named("nitr") = nitr);
  return List::create(Named("out") = wrap(X));
}


// [[Rcpp::export]]
SEXP ccorig_conv(
    const Eigen::MatrixXd & S,
    const Eigen::MatrixXd & X0_,
    const Eigen::MatrixXd & LambdaMat,
    double lambda2,
    double epstol,
    int maxitr
)
{
  Eigen::MatrixXd X = X0_;
  //Eigen::VectorXd maxd(maxitr);

  ccorig(S, X, LambdaMat, lambda2, epstol,
         maxitr);

  //return List::create(Named("out") = wrap(X), Named("maxdiff") = wrap(maxd), Named("nitr") = nitr);
  return List::create(Named("out") = wrap(X));
}



