\name{genmat}
\alias{genmat}
\alias{genmat.bed}
\alias{genmat.vcf}
\alias{genmat.gds}
\title{Get genotype matrix from genomic data
}
\description{Get genotype matrix from genomic data
}
\usage{
genmat(genfile, sample.id = NULL, snp.id = NULL, 
snpfirstdim = NA, .snpread = NA, with.id = FALSE, verbose = TRUE, ...)


## S3 method for class 'genmat.bed'
genmat.bed(genfile, sample.id = NULL, snp.id = NULL, 
snpfirstdim = NA, .snpread = NA, with.id = FALSE, verbose = TRUE, ...)



## S3 method for class 'genmat.vcf'
genmat.vcf(genfile, sample.id = NULL, snp.id = NULL, 
snpfirstdim = NA, .snpread = NA, with.id = FALSE, verbose = TRUE, ...)



## S3 method for class 'genmat.gds'
genmat.gds(genfile, sample.id = NULL, snp.id = NULL, 
snpfirstdim = NA, .snpread = NA, with.id = FALSE, verbose = TRUE, ...)


}

\arguments{
  \item{genfile}{Genetic datasets containg sample ID and SNP ID, format includes bed (plink), vcf, or GDS file.
}
  \item{sample.id}{Sample ID

}
  \item{snp.id}{SNP ID

}
  \item{snpfirstdim}{whether genotypes are stored in the individual-major mode (TRUE), (i.e, list all SNPs for the first individual, and then list all SNPs for the second individual, etc) or (FALSE) for snp-major mode; if NA, determine automatically

}
  \item{.snpread}{internal use

}
  \item{with.id}{whether return "sample.id" and "snp.id".

}
  \item{verbose}{whether printing information

}
  \item{\dots}{more arguments

}
}
\details{Effectively get genoptype matrix from various genotype formats, including bed, vcf, or gds.

}
\value{The function returns an integer matrix with values 0, 1, 2 or NA representing the number of reference allele when with.id=FALSE; or list(genotype, sample.id, snp.id) when with.id=TRUE. The orders of sample and SNP IDs in the genotype matrix are actually consistent with sample.id and snp.id in the GDS file, which may not be as the same as the arguments sampel.id and snp.id specified by users.

}
\references{
Zheng, X., & Weir, B. S. (2016). Eigenanalysis of SNP data with an identity by descent interpretation. Theoretical population biology, 107, 65-76.

}

\examples{

\donttest{inp=SNPRelate::snpgdsExampleFileName()

genomat1=genmat.gds(inp)}

}

\keyword{ Genotype matrix}

