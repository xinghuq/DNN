/*
 */

/** \file export.h
 *  \brief Exporting networks.
 */


#ifndef DNN_EXPORT_H

#define DNN_EXPORT_H


#include <vector>
#include <string>


namespace DNN {
namespace internal {


/// Export trained network to a C function with optional input and output
/// affine transformations of the form \f$Ax+b\f$ (input)
/// and \f$Cx+d\f$ (output). When with_wp is set to true the backpropagation code
/// (for online learning) is also exported. In order to export the backpropagation
/// code, when output transformation \f$Cx+d\f$ is provided, one has to provide
/// the inverse transformation given by \f$Ex+f\f$.
template <typename T>
bool mlp_export_C(const std::string &fname,
                  const std::string &netname,
                  const std::vector<int> &layers,
                  const std::vector<int> &n_p,
                  const std::vector<T> &w_val,
                  const std::vector<int> &w_fl,
                  int w_on,
                  const std::vector<int> &af,
                  const std::vector<T> &af_p,
                  bool with_bp,
                  const T *A, const T *b,
                  const T *C, const T *d,
                  const T *E, const T *f);



} /* namespace internal */
} /* namespace DNN */


#endif /* DNN_EXPORT_H */
