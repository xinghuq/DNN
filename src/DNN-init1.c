#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .C calls */
extern void ihessupdate(void *, void *, void *, void *, void *);
extern void mlp_eval(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mlp_grad(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mlp_gradi(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mlp_gradij(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mlp_jacob(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mlp_mse(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mlp_set_active(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

/* .Call calls */
extern SEXP actvfuncstr(SEXP);
extern SEXP DNN_ver();
extern SEXP mlp_construct(SEXP);
extern SEXP mlp_expand_reorder_inputs(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mlp_export(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mlp_export_C(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mlp_get_abs_w_idx(SEXP, SEXP);
extern SEXP mlp_import(SEXP);
extern SEXP mlp_merge(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mlp_rm_input_neurons(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mlp_rm_neurons(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mlp_stack(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP read_DNNdataset(SEXP);
extern SEXP write_DNNdataset(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CMethodDef CEntries[] = {
    {"ihessupdate",    (DL_FUNC) &ihessupdate,     5},
    {"mlp_eval",       (DL_FUNC) &mlp_eval,        9},
    {"mlp_grad",       (DL_FUNC) &mlp_grad,       12},
    {"mlp_gradi",      (DL_FUNC) &mlp_gradi,      13},
    {"mlp_gradij",     (DL_FUNC) &mlp_gradij,     13},
    {"mlp_jacob",      (DL_FUNC) &mlp_jacob,      13},
    {"mlp_mse",        (DL_FUNC) &mlp_mse,        10},
    {"mlp_set_active", (DL_FUNC) &mlp_set_active, 11},
    {NULL, NULL, 0}
};

static const R_CallMethodDef CallEntries[] = {
    {"actvfuncstr",               (DL_FUNC) &actvfuncstr,                1},
    {"DNN_ver",                   (DL_FUNC) &DNN_ver,                    0},
    {"mlp_construct",             (DL_FUNC) &mlp_construct,              1},
    {"mlp_expand_reorder_inputs", (DL_FUNC) &mlp_expand_reorder_inputs,  9},
    {"mlp_export",                (DL_FUNC) &mlp_export,                 7},
    {"mlp_export_C",              (DL_FUNC) &mlp_export_C,              16},
    {"mlp_get_abs_w_idx",         (DL_FUNC) &mlp_get_abs_w_idx,          2},
    {"mlp_import",                (DL_FUNC) &mlp_import,                 1},
    {"mlp_merge",                 (DL_FUNC) &mlp_merge,                  9},
    {"mlp_rm_input_neurons",      (DL_FUNC) &mlp_rm_input_neurons,       8},
    {"mlp_rm_neurons",            (DL_FUNC) &mlp_rm_neurons,            11},
    {"mlp_stack",                 (DL_FUNC) &mlp_stack,                  8},
    {"read_DNNdataset",           (DL_FUNC) &read_DNNdataset,            1},
    {"write_DNNdataset",          (DL_FUNC) &write_DNNdataset,           7},
    {NULL, NULL, 0}
};

void R_init_DNN(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
