#include <RcppEigen.h>

void ccista(

    const Eigen::MatrixXd S,
    Eigen::SparseMatrix<double> &X,
    const Eigen::MatrixXd LambdaMat,
    double lambda2,
    double epstol,
    int    maxitr,
    int    steptype
);

void ccfista(

    const Eigen::MatrixXd S,
    Eigen::SparseMatrix<double> &X,
    const Eigen::MatrixXd LambdaMat,
    double lambda2,
    double epstol,
    int maxitr,
    int steptype
);

void ccorig(

    const Eigen::MatrixXd S,
    Eigen::MatrixXd &X,
    const Eigen::MatrixXd LambdaMat,
    double lambda2,
    double epstol,
    int maxitr
);
