#ifndef TEST1_H
#define TEST1_H

#include "Eigen/Core"
#include "eigency.h"

void function_w_mat_arg(
    Eigen::Map<Eigen::MatrixXd> &mat,
    Eigen::Map<Eigen::VectorXd> &vec,
    double epstol
);

#endif
