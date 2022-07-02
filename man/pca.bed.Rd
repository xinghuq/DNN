\name{pca}
\alias{pca}
\alias{pca.bed}
\alias{pca.vcf}
\alias{pca.gds}
\title{ Principal Component Analysis (PCA) on whole genome data  with 

}
\description{Fast implementation of Principal Component Analysis (PCA) on whole genome data

}
\usage{
pca(genfile, sample.id = NULL, snp.id = NULL, autosome.only = TRUE, 
remove.monosnp = TRUE, maf = NaN, missing.rate = NaN, 
algorithm = c("exact", "randomized"), 
eigen.cnt = ifelse(identical(algorithm, "randomized"), 16L, 32L),
num.thread = 1L, bayesian = FALSE, need.genmat = FALSE, 
genmat.only = FALSE, eigen.method = c("DSPEVX", "DSPEV"), 
aux.dim = eigen.cnt * 2L, iter.num = 10L, verbose = TRUE,...)

## S3 method for class 'pca.bed'
pca.bed(genfile, sample.id = NULL, snp.id = NULL, autosome.only = TRUE, 
remove.monosnp = TRUE, maf = NaN, missing.rate = NaN, 
algorithm = c("exact", "randomized"), 
eigen.cnt = ifelse(identical(algorithm, "randomized"), 16L, 32L),
num.thread = 1L, bayesian = FALSE, need.genmat = FALSE, 
genmat.only = FALSE, eigen.method = c("DSPEVX", "DSPEV"), 
aux.dim = eigen.cnt * 2L, iter.num = 10L, verbose = TRUE,...)

## S3 method for class 'pca.vcf'
pca.vcf(genfile, sample.id = NULL, snp.id = NULL, autosome.only = TRUE, 
remove.monosnp = TRUE, maf = NaN, missing.rate = NaN, 
algorithm = c("exact", "randomized"), 
eigen.cnt = ifelse(identical(algorithm, "randomized"), 16L, 32L),
num.thread = 1L, bayesian = FALSE, need.genmat = FALSE, 
genmat.only = FALSE, eigen.method = c("DSPEVX", "DSPEV"), 
aux.dim = eigen.cnt * 2L, iter.num = 10L, verbose = TRUE,...)

## S3 method for class 'pca.gds'
pca.gds(genfile, sample.id = NULL, snp.id = NULL, autosome.only = TRUE, 
remove.monosnp = TRUE, maf = NaN, missing.rate = NaN, 
algorithm = c("exact", "randomized"), 
eigen.cnt = ifelse(identical(algorithm, "randomized"), 16L, 32L),
num.thread = 1L, bayesian = FALSE, need.genmat = FALSE, 
genmat.only = FALSE, eigen.method = c("DSPEVX", "DSPEV"), 
aux.dim = eigen.cnt * 2L, iter.num = 10L, verbose = TRUE,...)

}

\arguments{
  \item{genfile}{ Genetic datasets containg sample ID and SNP ID, format includes bed (plink), vcf, or GDS file.

}
  \item{sample.id}{a vector of sample id specifying selected samples; if NULL, all samples are used

}
  \item{snp.id}{a vector of snp id specifying selected SNPs; if NULL, all SNPs are used

}
  \item{autosome.only}{use autosomal SNPs only; if it is a numeric or character value, keep SNPs according to the specified chromosome.

}
  \item{remove.monosnp}{remove monomorphic SNPs

}
  \item{maf}{filter SNPs with ">= maf" only; if NaN, no MAF threshold

}
  \item{missing.rate}{filter the SNPs with "<= missing.rate" only; if NaN, no missing threshold

}
  \item{algorithm}{"exact", traditional exact calculation; "randomized", fast PCA with randomized algorithm introduced in Galinsky et al. 2016

}
  \item{eigen.cnt}{output the number of eigenvectors; if eigen.cnt <= 0, then return all eigenvectors

}
  \item{num.thread}{the number of (CPU) cores used; if NA, detect the number of cores automatically

}
  \item{bayesian}{if TRUE, use bayesian normalization

}
  \item{need.genmat}{if TRUE, return the genetic covariance matrix

}
  \item{genmat.only}{return the genetic covariance matrix only, do not compute the eigenvalues and eigenvectors

}
  \item{eigen.method}{"DSPEVX" -compute the top eigen.cnt eigenvalues and eigenvectors using LAPACK::DSPEVX; "DSPEV" -to be compatible with SNPRelate_1.1.6 or earlier, using LAPACK::DSPEV; "DSPEVX" is significantly faster than "DSPEV" if only top principal components are of interest

}
  \item{aux.dim}{auxiliary dimension used in fast randomized algorithm

}
  \item{iter.num}{iteration number used in fast randomized algorithm

}
  \item{verbose}{if TRUE, show information

}
  \item{...}{more arguments

}
}
\details{Efficient and fast implementation of PCA leveraging the advantage of Genomic Data Structure (GDS) to accelerate computations on SNP data using parallel computing for multi-core symmetric multiprocessing computer architectures. The minor allele frequency and missing rate for each SNP passed in snp.id are calculated over all the samples in sample.id.

}
\value{Return a of of PCA results, including sample id, SNP id and PCs.

\item{eigenval}{eigenvalues}
\item{eigenvect}{eigenvactors, "# of samples" x "eigen.cnt"}
\item{varprop}{variance proportion for each principal component}
}
\references{

Zheng, X., Weir, B. S. (2016). Eigenanalysis of SNP data with an identity by descent interpretation. Theoretical population biology, 107, 65-76.


Patterson N, Price AL, Reich D. (2006). Population structure and eigenanalysis. PLoS Genet.2(12):e190.

Galinsky KJ, Bhatia G, Loh PR, Georgiev S, Mukherjee S, Patterson NJ, Price AL. (2016). Fast Principal-Component Analysis Reveals Convergent Evolution of ADH1B in Europe and East Asia. Am J Hum Genet. 2016 Mar 3;98(3):456-72. 

}
\examples{

\donttest{
inp=SNPRelate::snpgdsExampleFileName()

pca1=pca.gds(inp, autosome.only=TRUE, remove.monosnp=TRUE, maf=0.01, missing.rate=0.1)
}}

\keyword{ PCA}

