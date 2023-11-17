# Title: func_data4ML_class.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script contains functions to pre-process data prior to ML modelling


process_data4ML <- function(EXPRS.FILE, PHENO.FILE, features_varname, OUT_DIR_DATA, target, myseed, export) {

  # Data load --------------------------------------------------------------------
  # star raw counts & (pre-filtered) clinical data.
  pheno <- vroom(PHENO.FILE, col_types = c("ccffdfildddddddddlf")) 
  expr <- vroom(EXPRS.FILE, col_types = cols()) %>%
    rename_with(toupper)
  expr = expr[!duplicated(expr[[features_varname]]),]
  
  
  # Data reformatting for ML -----------------------------------------------------
  expr_4ML <- expr %>%
    pivot_longer(cols = stringr::str_subset(colnames(expr), "[0-9]{4}\\.[A-Z]"), names_to = "SAMPLE", values_to = "EXPRS") %>%
    separate(SAMPLE, c("PATIENT_ID", "VISIT"), sep = '\\.') %>%
    pivot_wider(names_from = all_of(features_varname), values_from = EXPRS) %>%
    dplyr::select(-VISIT)
  
  pheno_4ML <- pheno %>%
    dplyr::select(all_of(c(target, 'PATIENT_ID'))) 
  
  expr_4ML <- expr_4ML %>%
    inner_join(pheno_4ML, 
               by = "PATIENT_ID") %>%
    mutate_at(target, factor)
  rm(pheno_4ML)
  
  
  # apply unsupervised filters ---------------------------------------------------
  # remove zero variance features 
  nzv <- nearZeroVar(expr_4ML[, !(names(expr_4ML) %in% c(target))], freqCut = 10 ) 
  if (length(nzv) != 0 ) {
    expr_4ML <- expr_4ML[,-nzv]
  }
  print(paste("nzv filter:", length(nzv)))
  
  # remove highly correlated features 
  cor_df = cor(expr_4ML[,-c(1,ncol(expr_4ML))]) # remove patient ID and diagnosis variables
  hc = findCorrelation(cor_df, cutoff=0.85, names = TRUE) 
  hc = sort(hc)
  print(paste("hc filter:", length(hc)))
  expr_4ML = expr_4ML[,-which(names(expr_4ML) %in% c(hc))]
  print("NZV and correlation filters successfully applied")
  print(paste((length(hc) + length(nzv)), "features were removed out of", nrow(expr)))
  
  expr_4ML <- expr_4ML %>% 
    column_to_rownames("PATIENT_ID")
  
  # export pre-processed data 
  if (export == TRUE) {
    if (e_level == "GENE") {
      readr::write_tsv(expr_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0("data_expr_4ML_", target, ".tsv")))
    } else {
      readr::write_tsv(expr_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0(e_level, "_", st, "_data_expr_4ML_", target, ".tsv")))
    }
  }
  

  # create training/held-out set -------------------------------------------------
  set.seed(myseed-1)
  inTraining <- createDataPartition(expr_4ML[[as.character(target)]], p = .85, list = FALSE)
  hout_4ML  <- expr_4ML[-inTraining, ]
  expr_4ML <- expr_4ML[inTraining, ] 
  # export pre-processed data split
  if (export == TRUE) {
    if (e_level == "GENE") {
      readr::write_tsv(expr_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0("data_cv_expr_4ML_", target, ".tsv")))
      readr::write_tsv(hout_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0("data_test_expr_4ML_", target, ".tsv")))
    } else {
      readr::write_tsv(expr_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0(e_level, "_", st, "_data_cv_expr_4ML_", target, ".tsv")))
      readr::write_tsv(hout_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0(e_level, "_", st, "_data_test_expr_4ML_", target, ".tsv")))
    }
  }
}