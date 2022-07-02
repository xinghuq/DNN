/*
 */

/** \file DNNcfg.R.h
 *  \brief DNN R configuration file.
 */

#ifndef DNNCFG_R_H

#define DNNCFG_R_H

#undef F77_DUMMY_MAIN
#undef F77_FUNC
#undef F77_FUNC_
#undef FC_DUMMY_MAIN_EQ_F77
#undef HAVE_BLAS
#undef HAVE_DLFCN_H
#undef HAVE_GETTIMEOFDAY
#undef HAVE_INTTYPES_H
#undef HAVE_LAPACK
#undef HAVE_LIBM
#undef HAVE_MEMORY_H
#undef HAVE_OPENMP
#undef HAVE_STDINT_H
#undef HAVE_STDLIB_H
#undef HAVE_STRINGS_H
#undef HAVE_STRING_H
#undef HAVE_SYS_STAT_H
#undef HAVE_SYS_TIME_H
#undef HAVE_SYS_TYPES_H
#undef HAVE_UNISTD_H
#undef LT_OBJDIR
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION
#undef STDC_HEADERS

#define HAVE_BLAS 1
#include <R_ext/RS.h>
#define F77_FUNC(name,NAME) F77_NAME(name)

#include <Rconfig.h>
#if defined(_OPENMP) || defined(SUPPORT_OPENMP)
#define HAVE_OPENMP 1
#endif /* defined(_OPENMP) || defined(SUPPORT_OPENMP) */

#endif /* DNNCFG_R_H */
