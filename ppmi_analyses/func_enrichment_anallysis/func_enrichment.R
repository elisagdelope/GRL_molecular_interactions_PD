# Title: func_enrichment.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script contains functions to perform enrichment analysis on different databases (kegg, go, mesh)

library("limma")


options(timeout = 3600)
gsea_kegg <- function(de, fdr=0.05) {
  if (length(de) > 0) {
    topHit <- limma::kegga(de) 
    topHit <- topHit[topHit$P.DE < fdr,] # P.DE is p-value for over-representation of the KEGG pathway in the set
    topHit <- topKEGG(topHit, number=Inf) # sort by p-value, extract all KEGG pathways from kegga output
    topHit <- topHit[, c("Pathway", "P.DE", "N", "DE")] # N is the number of genes in the KEGG pathway; DE is the number of genes in the DE set
    if (all(is.na(topHit[["Pathway"]]))){
      PathID.PathName.hsa <- limma::getKEGGPathwayNames("hsa", remove=TRUE)
      m <- match(rownames(topHit), paste0("path:",PathID.PathName.hsa[,1]))
      topHit$Pathway <- PathID.PathName.hsa[m,2]
    }
    return(topHit)
  }
  return(NA)
}

gsea_go <- function(de, fdr=0.05) {
  if (length(de) > 0) {
    topHit <- limma::goana(de)
    topHit <- topHit[topHit$P.DE < fdr,]
    topHit <- topGO(topHit, number=Inf)
    topHit <- topHit[, c("Term", "Ont", "P.DE", "N", "DE")]
    return(topHit)
  }
  return(NA)
}

mesh_enrich <- function(de, genes, target_meshterms=c("Parkinson Disease", "Parkinson Disease, Secondary", "Parkinsonian Disorders"), params) {
  meshR <- meshHyperGTest(params)
  meshR <- summary(meshR)
  meshR$GENEID <- NULL
  meshR$SOURCEID <- NULL
  meshR <- meshR[!duplicated(meshR), ]
  
  if (nrow(meshR) > 0) {
    meshR$Rank <- 1:nrow(meshR) # Rank is the actual rank with respect to other diseases (e.g. ranked 5th)
    meshR$Rank_p <- 1:nrow(meshR) / nrow(meshR) # rank_p is the normalized rank, i.e. rank in the top n%
  }
  
  meshR <- meshR[meshR$MESHTERM %in% target_meshterms, ]
  if (params@pAdjust == "none") {
    meshR <- meshR[, c("MESHTERM", "Pvalue", "ExpCount", "Count", "Rank", "Rank_p")] # count is number of genes mapping to the disease ; expected number of genes mapping under the null hypothesis of no enrichment for that disease
  } else {
    names(meshR)[names(meshR) == "BH"] <- "FDR"
    meshR <- meshR[, c("MESHTERM", "Pvalue", "FDR", "ExpCount", "Count", "Rank", "Rank_p")]
  }
  
  return(meshR)
}