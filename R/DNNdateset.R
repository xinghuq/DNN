


#' Reading and writing datasets in the DNN format
#'
#' These functions can be used to read and write datasets from/to a text file
#' in the DNN format. Datasets in the similar  format (comments are not
#' supported by ) can also be read by \code{read.DNNdataset}.
#'
#' Files are organised as follows:
#' \itemize{
#'  \item The first comment (beginning with \code{#}) is the dataset information (ignored on read),
#'  \item three numbers determine: number of records, no. of inputs and no. of outputs,
#'  \item each data record has two or three lines:
#'     \itemize{
#'      \item (optional) record information in a comment (beginning with \code{#}),
#'      \item line with input values,
#'      \item line with output values.
#'      }
#'  }
#'
#' @param fname character string with the filename
#' @param input numeric matrix, each row corresponds to one input vector
#' @param output numeric matrix with rows corresponding to expected outputs,
#'        the number of rows must be equal to the number of input rows
#'
#' @return \code{read.DNNdataset} returns a dataframe.
#'
#'         \code{write.DNNdataset} does not return.
#'
#' @examples
#'
#' # set up the XOR problem inputs and outputs
#' inp <- c(0, 0, 1, 1, 0, 1, 0, 1)
#' dim(inp) <- c(4, 2)
#' outp <- c(0, 1, 1, 0)
#' dim(outp) <- c(4, 1)
#' # write dataset
#' write.DNNdataset("xor.dat", inp, outp)
#' # show the output file
#' file.show("xor.dat")
#' # read dataset
#' xordf <- read.DNNdataset("xor.dat")
#' # show the imported dataset
#' show(xordf)
#'
#' @name read-write-DNNdataset
#'
#' @export
#'
read.DNNdataset <- function(fname)
{
    if (!is.character(fname) || (length(fname) != 1)) {
        stop("invalid filename")
    }
    ret <- .Call("read_DNNdataset",fname)
    if (is.null(ret)) stop("reading failed");
    R <- ret[[1]]
    CI <- ret[[2]]
    CO <- ret[[3]]
    rinf <- ret[[4]]
    inp <- ret[[5]]
    outp <- ret[[6]]
    dim(inp) <- c(CI, R)
    inp <- t(inp)
    dim(outp) <- c(CO, R)
    outp <- t(outp)
    res <- cbind(inp, outp)
    if (all(rinf == "")) rinf <- as.character(1:R)
    drinf <- duplicated(rinf)
    if (any(drinf)) {
        warning("duplicated record information")
        rinf[which(drinf)] <- paste0(rinf[which(drinf)], " (", which(drinf), ")")
    }
    rownames(res) <- rinf
    colnames(res) <- c(paste0("in ", 1:CI), paste0("out ", 1:CO))
    return (as.data.frame(res))
}


#' @rdname read-write-DNNdataset
#'
#' @export
#'
write.DNNdataset <- function(fname, input, output)
{
    if (!is.character(fname) || (length(fname) != 1)) {
        stop("invalid filename")
    }
    if (!is.numeric(input)) {
        stop("invalid input, expected numeric matrix")
    }
    di <- dim(input)
    if (length(di) != 2) {
        stop("invalid input, expected numeric matrix")
    }
    R <- di[1]
    CI <- di[2]
    if (!is.numeric(output)) {
        stop("invalid output, expected numeric matrix")
    }
    do <- dim(output)
    if (length(do) != 2) {
        stop("invalid output, expected numeric matrix")
    }
    CO <- do[2]
    if (do[1] != R) {
        stop("no. of output rows and no. of input rows disagree")
    }
    if (R == 0) {
        stop("no data records")
    }
    rinfo <- rownames(input)
    if (is.null(rinfo)) rinfo <- rownames(output);
    if (is.null(rinfo)) rinfo <- rep("", R)
    ok <- .Call("write_DNNdataset",
                fname,
                as.integer(R), as.integer(CI), as.integer(CO),
                rinfo, as.numeric(input), as.numeric(output))
    if (!ok) stop("writing failed");
}

