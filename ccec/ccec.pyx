# distutils: language = c++
# distutils: sources = ccec/ccec_cpp.cpp

from eigency.core cimport *

cdef extern from "ccec/ccec_cpp.h":

     cdef void _function_w_mat_arg "function_w_mat_arg"(Map[MatrixXd] &, Map[VectorXd] &, double)

# Function with matrix argument.
def function_w_mat_arg(np.ndarray[np.float64_t, ndim=2] array2d, np.ndarray[np.float64_t, ndim=1] array1d, double epstol):
    return _function_w_mat_arg(Map[MatrixXd](array2d), Map[VectorXd](array1d), epstol)
