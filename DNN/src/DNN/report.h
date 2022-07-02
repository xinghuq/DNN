/*
 */

/** \file report.h
 * \brief Reporting.
 */

#ifndef DNN_REPORT_H

#define DNN_REPORT_H

#include <string>


namespace DNN {
namespace internal {


/// Report, e.g. teaching progress (on std::cout, or somewhere else).
void report(const std::string &);


} /* namespace internal */
} /* namespace DNN */

#endif /* DNN_UTILS_H */
