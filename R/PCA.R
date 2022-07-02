## Performing PCA for genomic datasets, bed,vcf, gds
"pca"=function(genfile, sample.id=NULL, snp.id=NULL,
               autosome.only=TRUE, remove.monosnp=TRUE, maf=NaN, missing.rate=NaN,
               algorithm=c("exact", "randomized"),
               eigen.cnt=ifelse(identical(algorithm, "randomized"), 16L, 32L),
               num.thread=1L, bayesian=FALSE, need.genmat=FALSE,
               genmat.only=FALSE, eigen.method=c("DSPEVX", "DSPEV"),
               aux.dim=eigen.cnt*2L, iter.num=10L, verbose=TRUE, ...){
  UseMethod("pca")
}

pca.bed=function(genfile, sample.id=NULL, snp.id=NULL,
                 autosome.only=TRUE, remove.monosnp=TRUE, maf=NaN, missing.rate=NaN,
                 algorithm=c("exact", "randomized"),
                 eigen.cnt=ifelse(identical(algorithm, "randomized"), 16L, 32L),
                 num.thread=1L, bayesian=FALSE, need.genmat=FALSE,
                 genmat.only=FALSE, eigen.method=c("DSPEVX", "DSPEV"),
                 aux.dim=eigen.cnt*2L, iter.num=10L, verbose=TRUE,...){

  SNPRelate::snpgdsBED2GDS(paste0(genfile, "bed"), paste0(genfile, "fam"), paste0(genfile, "bim"), "inputgenofile.gds")
  genf = SNPRelate::snpgdsOpen("inputgenofile.gds")
  PCs=SNPRelate::snpgdsPCA(genf, sample.id=sample.id, snp.id=snp.id,
            autosome.only=autosome.only, remove.monosnp=remove.monosnp, maf=maf, missing.rate=missing.rate,
            algorithm=algorithm,
            eigen.cnt=eigen.cnt,
            num.thread=num.thread, bayesian=bayesian, need.genmat=need.genmat,
            genmat.only=genmat.only, eigen.method=eigen.method,
            aux.dim=aux.dim, iter.num=iter.num, verbose=verbose,...)

  SNPRelate::snpgdsClose(genf)
  unlink("inputgenofile.gds", force = TRUE)
return(list(PCs,class = "pca"))
}

pca.vcf=function(genfile, sample.id=NULL, snp.id=NULL,
                 autosome.only=TRUE, remove.monosnp=TRUE, maf=NaN, missing.rate=NaN,
                 algorithm=c("exact", "randomized"),
                 eigen.cnt=ifelse(identical(algorithm, "randomized"), 16L, 32L),
                 num.thread=1L, bayesian=FALSE, need.genmat=FALSE,
                 genmat.only=FALSE, eigen.method=c("DSPEVX", "DSPEV"),
                 aux.dim=eigen.cnt*2L, iter.num=10L, verbose=TRUE,...){
  
  SNPRelate::snpgdsVCF2GDS(genfile, "inputgenofile.gds", method = "biallelic.only")
  genf = SNPRelate::snpgdsOpen("inputgenofile.gds")
  PCs=snpgdsPCA(genf, sample.id=sample.id, snp.id=snp.id,
                autosome.only=autosome.only, remove.monosnp=remove.monosnp, maf=maf, missing.rate=missing.rate,
                algorithm=algorithm,
                eigen.cnt=eigen.cnt,
                num.thread=num.thread, bayesian=bayesian, need.genmat=need.genmat,
                genmat.only=genmat.only, eigen.method=eigen.method,
                aux.dim=aux.dim, iter.num=iter.num, verbose=verbose,...)
  
  SNPRelate::snpgdsClose(genf)
  unlink("inputgenofile.gds", force = TRUE)
  return(list(PCs,class = "pca"))
}


pca.gds=function(genfile, sample.id=NULL, snp.id=NULL,
                 autosome.only=TRUE, remove.monosnp=TRUE, maf=NaN, missing.rate=NaN,
                 algorithm=c("exact", "randomized"),
                 eigen.cnt=ifelse(identical(algorithm, "randomized"), 16L, 32L),
                 num.thread=1L, bayesian=FALSE, need.genmat=FALSE,
                 genmat.only=FALSE, eigen.method=c("DSPEVX", "DSPEV"),
                 aux.dim=eigen.cnt*2L, iter.num=10L, verbose=TRUE,...){
  genf = SNPRelate::snpgdsOpen(genfile)
  PCs=snpgdsPCA(genf, sample.id=sample.id, snp.id=snp.id,
                autosome.only=autosome.only, remove.monosnp=remove.monosnp, maf=maf, missing.rate=missing.rate,
                algorithm=algorithm,
                eigen.cnt=eigen.cnt,
                num.thread=num.thread, bayesian=bayesian, need.genmat=need.genmat,
                genmat.only=genmat.only, eigen.method=eigen.method,
                aux.dim=aux.dim, iter.num=iter.num, verbose=verbose,...)
  
  
  return(list(PCs,class = "pca"))
}





