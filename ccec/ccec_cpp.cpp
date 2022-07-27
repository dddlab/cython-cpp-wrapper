#include "ccec_cpp.h"

#include <iostream>

void function_w_mat_arg(Eigen::Map<Eigen::MatrixXd> &mat, double epstol) {
    mat(0,0) = epstol;
}
