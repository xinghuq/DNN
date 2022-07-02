/*
 */

/** \file error.h
 *  \brief Class DNN::exception and error handling.
 */


#ifndef DNN_ERROR_H

#define DNN_ERROR_H


#include <stdexcept>
#include <string>
#ifdef DNN_DEBUG
#include <iostream>
#include <cstdlib>
#endif


namespace DNN {

/// Exception class for handling errors in DNN.
class exception : public std::exception {
  public:
    /// Constructor
    explicit exception(const std::string &s)
        : m_mes(s) { ; }
    /// Destructor
    virtual ~exception() throw() { ; }
    /// C string with error message
    virtual const char* what() const throw() { return m_mes.c_str(); }

  private:
    /// Error message.
    std::string m_mes;

}; /* class exception */



/// Error handling
inline
void
error(const std::string &s)
{
#ifdef DNN_DEBUG
    std::cerr << "DNN error: " << s << "\naborting...\n";
    abort();
#else
    throw DNN::exception(s);
#endif
}



} /* namespace DNN */

#endif /* DNN_ERROR_H */
