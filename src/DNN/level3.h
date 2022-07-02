/*
 */

/** \file level3.h
 *  \brief Level 3 operations: evaluation, MSE, and gradients plus
 *  Hessian inverse update used in OBS.
 */

#ifndef DNN_LEVEL3_H

#define DNN_LEVEL3_H


namespace DNN {
namespace internal {


/// Evaluate network output given input
template <typename T>
void
eval(const int *lays, int no_lays, const int *n_pts,
     const T *w_val, const int *af, const T *af_p,
     int no_datarows, const T *in, T *out);


/// Determine network's MSE given input and expected output.
template <typename T>
T
mse(const int *lays, int no_lays, const int *n_pts,
    const T *w_val, const int *af, const T *af_p,
    int no_datarows, const T *in, const T *out);


/// Compute gradient of MSE (derivatives w.r.t. active weights)
/// given input and expected output.
template <typename T>
T
grad(const int *lays, int no_lays, const int *n_pts,
     const int *w_pts, const int *w_fl, const T *w_val,
     const int *af, const T *af_p,
     int no_datarows, const T *in, const T *out, T *gr);

/// Compute gradient of MSE (derivatives w.r.t. active weights)
/// given input and expected output using ith row of data only. This is
/// normalised by the number of outputs only.
template <typename T>
void
gradi(const int *lays, int no_lays, const int *n_pts,
      const int *w_pts, const int *w_fl, const T *w_val,
      const int *af, const T *af_p,
      int no_datarows, int i, const T *in, const T *out, T *gr);

/// Compute gradients of networks outputs, i.e the derivatives of outputs
/// w.r.t. active weights, at given data row.
template <typename T>
void
gradij(const int *lays, int no_lays, const int *n_pts,
       const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
       const int *af, const T *af_p,
       int no_datarows, int i, const T *in, T *gr);

/// Compute the Jacobian of network transformation, i.e the derivatives
/// of outputs w.r.t. network inputs, at given data row.
template <typename T>
void
jacob(const int *lays, int no_lays, const int *n_pts,
      const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
      const int *af, const T *af_p,
      int no_datarows, int i, const T *in, T *jac);

/// Update Hessian inverse approximation given result from gradij.
/// Implemented only if BLAS library is present.
template <typename T>
void
ihessupdate(int nw, int no, T a, const T *g, T *Hinv);



} /* namespace internal */
} /* namespace DNN */


#endif /* DNN_LEVEL3_H */
