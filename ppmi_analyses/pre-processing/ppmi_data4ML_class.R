# Title: ppmi_data4ML_class.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script applies pre-processing steps to generate data for ML modelling RNAseq snapshot data (BL) from gene expression and higher-level representations.
# Usage: Rscript ppmi_data4ML_class.R -n yes (for normalized counts)
# Data: data from expression counts at specific timepoint (BL), clinical data (e.g., diagnosis, age, sex).

# GC ---------------------------------------------------------------------------
rm(list = ls())
gc(T)



# Packages ---------------------------------------------------------------------
library(readr)
library(plyr)
library(dplyr)
library(vroom)
library(tidyr)
library(tibble)
library(stringr)
library(caret)
library(argparser, quietly = TRUE)
library(matrixStats)




# I/O --------------------------------------------------------------------------
# cmd line arguments
p <- arg_parser("PreprocessCV", hide.opts = FALSE)
p <- add_argument(parser = p, arg = "--analysis", help = "name of analysis directory (e.g., 02-pred-BL-PD, 02-pred-BL-UPDRS3-class)", required = TRUE)
p <- add_argument(parser = p, arg = "--norm", help = "boolean string (yes/no) whether to use deseq2-flt-normalized counts or not (raw counts)", default = "YES", type = "string", nargs = 1)
argv <- parse_args(p, argv = commandArgs(trailingOnly = TRUE))
norm_counts <- toupper(argv$norm)
analysis_name <-argv$analysis 
OUT_DIR <- paste0("../data/", analysis_name , "/02-outfiles") 
OUT_DIR_PATHWAY <- paste0("../data/", analysis_name , "/04-pathway_level") 
OUT_DIR_DATA <- paste0("../data/", analysis_name , "/05-data4ML")
myseed = 111

if (grepl("BL-PD", analysis_name)) {
  PHENO.FILE <- file.path(OUT_DIR, "pheno_BL.tsv")
  target = "DIAGNOSIS"
} else if (grepl("BL-UPDRS3", analysis_name)){
  PHENO.FILE <- file.path(OUT_DIR, "ppmi_pheno.tsv")
  target = "UPDRS3_binary"
}
source("func_data4ML_class.R")



# Main -------------------------------------------------------------------------
if ((!dir.exists(OUT_DIR)) | (!dir.exists(OUT_DIR_PATHWAY)) | (!dir.exists(OUT_DIR_DATA))) {
  dir.create(OUT_DIR, recursive = T)
  dir.create(OUT_DIR_PATHWAY, recursive = T)
  dir.create(OUT_DIR_DATA, recursive = T)
}

for (e_level in c("GENE", "GOBP", "GOCC", "CORUM")) {
  if ((e_level == "GENE") & (norm_counts == "NO")) { 
    EXPRS.FILE <- file.path(OUT_DIR, "flt_star_BL.tsv")
    features_varname <- "GENEID"
    process_data4ML(EXPRS.FILE, features_varname, OUT_DIR_DATA, OUT_DIR, target, myseed, export=TRUE)
  } else if ((e_level == "GENE") & (norm_counts == "YES")) { # normalized counts at gene level
    EXPRS.FILE <- file.path(OUT_DIR, "flt_norm_star_BL.tsv")
    features_varname <- "GENEID"
    process_data4ML(EXPRS.FILE, features_varname, OUT_DIR_DATA, OUT_DIR, target, myseed, export=TRUE)
  } else { # aggregationss
    
    for (st in c("mean", "median", "sd", "pca", "pathifier")) {
      EXPRS.FILE <- file.path(OUT_DIR_PATHWAY, paste(e_level, st, "expression.tsv", sep = "_"))
      features_varname <- paste0(e_level, "_NAME")
      
      if ((!e_level %in% c("GOBP", "GOCC", "CORUM")) | (!st %in% c("mean", "median", "sd", "pathifier", "pca")) | (!file.exists(EXPRS.FILE))) { 
        stop("Adequate arguments were not provided. Check R ppmi_rnaseq_binaryclass.R --help for the right usage.")
      }
      process_data4ML(EXPRS.FILE, features_varname, OUT_DIR_DATA, OUT_DIR, target, myseed, export=TRUE)
    }
  }
}    



# Session info -----------------------------------------------------------------
rm(list = ls())
gc(T)
cat("\n================\n  SESSION INFO\n================\n")
sessionInfo()


