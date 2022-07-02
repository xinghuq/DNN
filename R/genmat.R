## get genotype matrix from bed,vcf, gds
"genmat"=function(genfile, sample.id=NULL, snp.id=NULL, snpfirstdim=NA,
                  .snpread=NA, with.id=FALSE, verbose=TRUE, ...){
  UseMethod("genmat")
}


genmat.bed=function (genfile, sample.id=NULL, snp.id=NULL, snpfirstdim=NA,
                     .snpread=NA, with.id=FALSE, verbose=TRUE, ...) 
{
  SNPRelate::snpgdsBED2GDS(paste0(genfile, "bed"), paste0(genfile, "fam"), paste0(genfile, "bim"), "inputgenofile.gds")
  genf = SNPRelate::snpgdsOpen("inputgenofile.gds")
  genomat <- SNPRelate::snpgdsGetGeno(genf, sample.id=sample.id, snp.id=snp.id, snpfirstdim=snpfirstdim,
                                      .snpread=.snpread, with.id=with.id, verbose=verbose)
  
  SNPRelate::snpgdsClose(genf)
  unlink("inputgenofile.gds", force = TRUE)
  return(list(genomat = genomat, class = "genmat"))
}


genmat.vcf=function (genfile, sample.id=NULL, snp.id=NULL, snpfirstdim=NA,
                     .snpread=NA, with.id=FALSE, verbose=TRUE, ...) 
{
  SNPRelate::snpgdsVCF2GDS(genfile, "inputgenofile.gds", method = "biallelic.only")
  genf = SNPRelate::snpgdsOpen("inputgenofile.gds")
  genomat <- SNPRelate::snpgdsGetGeno(genf, sample.id=sample.id, snp.id=snp.id, snpfirstdim=snpfirstdim,
                                      .snpread=.snpread, with.id=with.id, verbose=verbose)
  
  SNPRelate::snpgdsClose(genf)
  unlink("inputgenofile.gds", force = TRUE)
  return(list(genomat = genomat, class = "genmat"))
}

genmat.gds=function(genfile, sample.id=NULL, snp.id=NULL, snpfirstdim=NA,
                    .snpread=NA, with.id=FALSE, verbose=TRUE,...) 
{
  genf = SNPRelate::snpgdsOpen(genfile)
  genomat <- SNPRelate::snpgdsGetGeno(genf, sample.id=sample.id, snp.id=snp.id, snpfirstdim=snpfirstdim,
                                      .snpread=.snpread, with.id=with.id, verbose=verbose)
  
  SNPRelate::snpgdsClose(genf)
  unlink("inputgenofile.gds", force = TRUE)
  return(list(genomat = genomat, class = "genmat"))
}



