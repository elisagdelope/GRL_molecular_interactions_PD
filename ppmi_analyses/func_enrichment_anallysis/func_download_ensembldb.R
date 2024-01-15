# Title: func_download_ensembldb.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script contains functions to download ensembl db data


### ------------------------------------------------ 
# Functions to download ensembl db data
# Ensembl db allows mapping to other databases
#
# usage:
# call download_ensembldb() to build db
# e.db <- download_ensembldb()
#
# call download_ensembldb_fromlist() to get the db info from gene list
# e.list.db <- download_ensembldb_fromlist(res.flt$gene_id)
#
# call download_ensembldb_fromlist() per batches if the list is too long
# ensembl_db <- DataFrame()
# for (i in seq(1, length(res.flt$gene_id), 500)) {
#   tg <- res.flt$gene_id[i:(i + 499)]
#   gene2ensembl <- download_ensembldb_fromlist(tg)
#   ensembl_db <- rbind(ensembl_db, gene2ensembl)
# }

library(tidyr)
library(dplyr)
library(biomaRt)


defineargs_ensembl <- function() {
  ensembl <- biomaRt::useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  target_attributes <- c("ensembl_gene_id",
                         "ensembl_gene_id_version",
                         "ensembl_transcript_id",
                         "ensembl_transcript_id_version",
                         "ensembl_peptide_id",
                         "ensembl_peptide_id_version",
                         "entrezgene_id", 
                         "entrezgene_accession", 
                         "refseq_mrna",
                         "wikigene_id", 
                         "wikigene_name", 
                         "wikigene_description",
                         "hgnc_id",
                         "hgnc_symbol",
                         "hgnc_trans_name",
                         "chromosome_name", 
                         "start_position", 
                         "end_position",
                         "transcript_gencode_basic",
                         "refseq_mrna_predicted",
                         "entrezgene_trans_name",
                         "embl",
                         "refseq_ncrna",
                         "refseq_ncrna_predicted",
                         "ucsc"
  )
  return(list(target_attributes, ensembl))
}

defineargs_ensembl_4annotation <- function() {
  ensembl <- biomaRt::useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  target_attributes <- c("ensembl_gene_id",
                         "ensembl_gene_id_version",
                         "ensembl_peptide_id",
                         "ensembl_peptide_id_version",
                         "entrezgene_accession", 
                         "wikigene_name", 
                         "wikigene_description",
                         "chromosome_name", 
                         "hgnc_symbol",
                         "embl",
                         "ucsc"
  )
  return(list(target_attributes, ensembl))
}
download_ensembldb_4annotation <- function() {
  message("Annotation variables from ensembl db will be downloaded")
  
  args_ensembl <- defineargs_ensembl_4annotation()
  attributes <- args_ensembl[[1]]
  ensembl_obj <- args_ensembl[[2]]
  eDB <- NULL
  
  for (i in seq(1, length(attributes), 3)) {
    tA <- attributes[i:(i + 2)]
    tA <- tA[!is.na(tA)]
    if (!"ensembl_transcript_id" %in% tA) {
      tAttr <- c("ensembl_transcript_id", tA)
    } else {
      tAttr <- tA
    }
    
    fDB <- biomaRt::getBM(mart = ensembl_obj, attributes = tAttr)
    
    if (!is.null(eDB)) {
      eDB <- merge(eDB, fDB, by = "ensembl_transcript_id")
    } else {
      eDB <- fDB
    }
  }
  
  OUTPUT_FILE <- paste0("../references/ensembldb_", Sys.Date(), ".tsv")
  readr::write_tsv(eDB, file = OUTPUT_FILE)
  message("Download complete")
  return(eDB)
}

download_ensembldb <- function() {
  message("A quasicomplete ensembl db will be downloaded")
  
  args_ensembl <- defineargs_ensembl()
  attributes <- args_ensembl[[1]]
  ensembl_obj <- args_ensembl[[2]]
  eDB <- NULL
  
  for (i in seq(1, length(attributes), 3)) {
    tA <- attributes[i:(i + 2)]
    tA <- tA[!is.na(tA)]
    if (!"ensembl_transcript_id" %in% tA) {
      tAttr <- c("ensembl_transcript_id", tA)
    } else {
      tAttr <- tA
    }
    
    fDB <- biomaRt::getBM(mart = ensembl_obj, attributes = tAttr)
    
    if (!is.null(eDB)) {
      eDB <- merge(eDB, fDB, by = "ensembl_transcript_id")
    } else {
      eDB <- fDB
    }
  }
  
  OUTPUT_FILE <- paste0("../references/ensembldb_", Sys.Date(), ".tsv")
  readr::write_tsv(eDB, file = OUTPUT_FILE)
  message("Download complete")
  return(eDB)
}





download_ensembldb_fromlist <- function(GENE_LIST) {
  if (!length(GENE_LIST) == 0) {
    message("Ensembl db of gene list will be downloaded")
  } else {
    message("###--------- ALERT: argument of type list is needed and has not been provided -> execution will stop -----------###")
    stop()
  }
  
  args_ensembl <- defineargs_ensembl()
  attributes <- args_ensembl[[1]]
  ensembl_obj <- args_ensembl[[2]]
  eDB <- NULL
  
  for (i in seq(1, length(attributes), 3)) {
    tA <- attributes[i:(i + 2)]
    tA <- tA[!is.na(tA)]
    if (!"ensembl_transcript_id" %in% tA) {
      tAttr <- c("ensembl_transcript_id", tA)
    } else {
      tAttr <- tA
    }
  
    fDB <- biomaRt::getBM(mart = ensembl_obj, attributes = tAttr, filters = "ensembl_gene_id", values = GENE_LIST)
    
    if (!is.null(eDB)) {
      eDB <- merge(eDB, fDB, by = "ensembl_transcript_id")
    } else {
      eDB <- fDB
    }
  }
  
  OUTPUT_FILE <- paste0("../references/ensembldb_fromlist_", Sys.Date(), ".tsv")
  readr::write_tsv(eDB, file = OUTPUT_FILE)
  message("Download complete")
  return(eDB)
}



# biomaRt::listAttributes(ensembl) # show list of available attributes
# fDB <- biomaRt::getBM(mart=ensembl, attributes=tAttr, filters="ensembl_gene_id", values = GENE_LIST) # downloads a given list of gene ids
# fDB <- biomaRt::getBM(mart = ensembl, attributes=tAttr) # supposedly downloads all but with some kind of restrictions (in reality not all genes are downloaded)



download_ids2ensembldb <- function() {
  
  gs2entrez = read.table('https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=md_eg_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit', 
                         sep = "\t", comment.char = "", quote = "", header = T)
  gs2ensembl = read.table('https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=md_ensembl_id&status=Approved&order_by=gd_app_sym_sort&format=text&submit=submit', 
                          sep = "\t", comment.char = "", quote = "", header = T)

  eDB <- gs2ensembl %>%
    full_join(gs2entrez,  by = "Approved.symbol") %>%
    rename(c( Ensembl.ID = "Ensembl.ID.supplied.by.Ensembl.", Entrez.ID = "NCBI.Gene.ID.supplied.by.NCBI."))

  OUTPUT_FILE <- paste0("../references/ids2ensembldb_", Sys.Date(), ".tsv")
  readr::write_tsv(eDB, file = OUTPUT_FILE)
  message("Download complete")
  return(eDB)
  
}



# other method to get entrezIDs from ensemblIDs: 
# r_sig$entrezid <- mapIds(org.Hs.eg.db,
#                          keys = r_sig$gene_id,
#                          column = "ENTREZID",
#                          keytype = "ENSEMBL",
#                          multiVals = "first")

