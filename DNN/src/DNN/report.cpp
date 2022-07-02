/*
 */

/** \file report.cpp
 * \brief Reporting.
 */


#include <DNN/report.h>
#ifdef R_SHAREDLIB
#include <R.h>
#else /* R_SHAREDLIB */
#include <iostream>
#endif /* R_SHAREDLIB */


void
DNN::internal::report(const std::string &s)
{
#ifdef R_SHAREDLIB
    Rprintf((s + '\n').c_str());
#else
    std::cout << s << '\n';
#endif
}












