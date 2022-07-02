/*
 */


/** \file utils.h
 * \brief Utility functions.
 */

#ifndef DNN_UTILS_H

#define DNN_UTILS_H

#include <string>
#include <vector>
#include <iostream>


namespace DNN {
namespace internal {



/// Messages.
class message {
public:
    /// Constructor
    message() { ; }
    /// Destructor
    ~message() { ; }

    /// Conversion to std::string.
    operator std::string() const { return m_mes; }

    /// Append char.
    message& operator<<(char);
    /// Append text.
    message& operator<<(const char*);
    /// Append text.
    message& operator<<(const std::string&);
    /// Append unsigned integer.
    message& operator<<(unsigned);
    /// Append integer.
    message& operator<<(int);
    /// Append float.
    message& operator<<(float);
    /// Append double.
    message& operator<<(double);

private:
    std::string m_mes;
}; /* class message */



/// Convert number to std::string.
std::string num2str(unsigned n);
/// Convert number to std::string.
std::string num2str(int n);
/// Convert number to std::string.
std::string num2str(float n);
/// Convert number to std::string.
std::string num2str(double n);

/// Return std::string with current date and time.
std::string time_str();

/// Return std::string with DNN version.
std::string DNN_ver();


/// Floating point precision.
template <typename T>
struct precision {
    static const int val = 0;
};

/// Floating point precision.
template <>
struct precision<float> {
    static const int val = 7;
};

/// Floating point precision.
template <>
struct precision<double> {
    static const int val = 16;
};


/// Read (int, float or double) from input stream.
template <typename T> bool read(std::istream& is, T &n);

/// Write comment to output stream.
bool write_comment(std::ostream&, const std::string&);

/// Read comment from output stream.
bool read_comment(std::istream &is, std::string &s);

/// Skip input until the end of comment.
void skip_comment(std::istream &is);

/// Skip whitespace and newlines.
void skip_blank(std::istream &is);

/// Skip whitespace, newlines and comments.
void skip_all(std::istream &is);

/// Are we at the end of line (whitespace ignored)?
bool is_eol(std::istream &is);

/// Are we at the end of line (whitespace ignored) followed by another eol
/// or a line beginning with comment?
bool is_deol(std::istream &is);

/// Are we at the end of line or file (whitespace ignored)?
bool is_eoleof(std::istream &is);

/// Draw a random sample of size M of integers from 1 to N.
std::vector<int> sample_int(int N, int M);


} /* namespace internal */
} /* namespace DNN */

#endif /* DNN_UTILS_H */
