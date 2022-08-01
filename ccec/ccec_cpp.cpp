#include "ccec_cpp.h"

#include <iostream>

void function_w_mat_arg(
    Eigen::Map<Eigen::MatrixXd> &mat, 
    Eigen::Map<Eigen::VectorXd> &vec, 
    double epstol) {
    mat(0,0) = epstol;
    vec(0) = 2.0 * epstol;
}
