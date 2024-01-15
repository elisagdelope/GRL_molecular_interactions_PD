# Title: ppmi_enrichment_grl_results.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script performs functional enrichment analysis (ORA) on the results of relevant genes identified by grl modeling pipelines (PPI and SSN)
# Usage: Rscript ppmi_enrichment_grl_results.R


# GC ---------------------------------------------------------------------------
rm(list = ls())
gc(T)



# Packages ---------------------------------------------------------------------
library(tidyr)
library(dplyr)
library(vroom)
library(stringr)
library(tibble)
library(meshr)
library(AnnotationHub)
library(MeSHDbi)
library(ggplot2)


# I/O --------------------------------------------------------------------------
source("func_download_ensembldb.R")
source("func_enrichment.R")
OUT_DIR_PPI <-  "../graph-analysis/rna-network/results/wandb/GAT_3p_20230928/cv-results/"
OUT_DIR_SSN <-  "../graph-analysis/patient-network/results/wandb/GAT_20230609/cv-results/"



# Load annotations & mesh terms ------------------------------------------------
ENSEMBL.FILE <- sort(list.files("../references/", pattern = "^ids2ensembldb_.+\\.tsv"), decreasing = TRUE)[1]  # take newest version
ids2ensembldb <- vroom(paste0("../references/", ENSEMBL.FILE), col_types = cols())
alpha = 0.05
ah <- AnnotationHub()
target_meshterms = c("Parkinson Disease", "Parkinson Disease, Secondary", "Parkinsonian Disorders", "Neurodegenerative Diseases")
ENRICH <- list()

translate_ensembl2entrez <- function(ensemble_list, ids2ensembldb){ # Translate Ensembl IDs to EntrezGene IDs
  ix <- match(ensemble_list, ids2ensembldb$Ensembl.ID)
  entrez_list <- ids2ensembldb$Entrez.ID[ix] #### ALERT: THERE CAN BE NAs - i.e. genes WITH NO ENTREZID
  print(paste0("Selected consistent genes (n = ", length(entrez_list), "; entrezID NAs = ", length(entrez_list[is.na(entrez_list)]), ")"))
  # remove NAs from list of genes
  entrez_list <- entrez_list[!is.na(entrez_list)]
  return(entrez_list)
} 



# -PPI--------------------------------------------------------------------------
# load all genes ppi
DATA.FILE <- "../graph-analysis/rna-network/data/data_rnaseq_ppi_4ML.csv"
data <- vroom(DATA.FILE, show_col_types = FALSE)
all_genes <- colnames(data)[-1]
all_genes_entrez <- translate_ensembl2entrez(all_genes, ids2ensembldb)


# load relevant genes in the ppi network
FILES <- c("cv_relevantnodes.csv", "cv_relevantedges_nodesdegree.csv")
for (f in FILES){
  genes <- vroom(file.path(OUT_DIR_PPI, f), col_types = cols(), show_col_types = FALSE)
  if (grepl("relevantnodes", f)) {
    genes_entrez <- translate_ensembl2entrez(genes$Feature, ids2ensembldb)
  } else if (grepl("relevantedges", f)) {
    genes_entrez <- translate_ensembl2entrez(genes$Node, ids2ensembldb)
  }
  ENRICH[["KEGG"]] <- gsea_kegg(genes_entrez)
  ENRICH[["GO"]] <- gsea_go(genes_entrez)
  meshParams <- new("MeSHHyperGParams", 
                    geneIds = genes_entrez, 
                    universeGeneIds = unique(all_genes_entrez),
                    annotation = "MeSH.Hsa.eg.db",
                    #  meshdb = "MeSH.db", # new requirement for BioC 3.14+
                    category = "C", 
                    database = "gendoo", 
                    pvalueCutoff = 1,
                    pAdjust = "BH")
  ENRICH[["MeSH"]] <- mesh_enrich(genes_entrez, all_genes, target_meshterms, meshParams)
  for (i in names(ENRICH)){
    if (!grepl("MeSH", i)) {
      ENRICH[[i]] <- ENRICH[[i]] %>% 
        as_tibble() %>%
        mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
        dplyr::filter(p.adjust < 0.05) %>%
        mutate(enrichment_scores = -log10(p.adjust)) %>%
        mutate(gene_ratio = DE /N ) 
    }
    readr::write_csv(ENRICH[[i]], file = file.path(OUT_DIR_PPI, paste0("enrichment_", i, "_", f)))
    print(knitr::kable(ENRICH[[i]][1:30,], row.names = F))
  }
  # plots
  # ORA - KEGG
  plot_name <- paste0("ENRICH_KEGG_", gsub('.{4}$', '', f), ".pdf")
  pdf(file = file.path(OUT_DIR_PPI, plot_name), width = 7, height = 7)
  ENRICH[["KEGG"]] %>%
    arrange(p.adjust) %>%
    slice_min(n=40, p.adjust) %>% 
    ggplot(aes(x = gene_ratio, y = reorder(Pathway, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (KEGG)") +
    theme_minimal() 
  
  dev.off()
  
  # ORA - GO
  topgo_p <- ENRICH[["GO"]] %>% 
    as_tibble() %>%
    mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
    dplyr::filter(DE > 1 & p.adjust < 0.05 & N >= 3) %>%
    mutate(enrichment_scores = -log10(P.DE)) %>%
    mutate(gene_ratio = DE /N ) %>%
    group_by(Ont) %>% 
    arrange(Ont, p.adjust, desc(gene_ratio)) %>%
    mutate(row_num = row_number()) %>%
    filter(row_num <= 10) %>%
    dplyr::select(-row_num)
  
  topgo_gr <- ENRICH[["GO"]] %>% 
    as_tibble() %>%
    mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
    dplyr::filter(DE > 1 & p.adjust < 0.05 & N >= 3) %>%
    mutate(enrichment_scores = -log10(P.DE)) %>%
    mutate(gene_ratio = DE /N ) %>%
    group_by(Ont) %>% 
    arrange(Ont, desc(gene_ratio), p.adjust) %>%
    mutate(row_num = row_number()) %>%
    filter(row_num <= 20) %>%
    dplyr::select(-row_num) %>%
    filter(Ont != "MF")
  
  pd_terms <- "Parkin|parkin|neuron|ubiquit|dopamine|carnitine|beta-oxidation|bile|T cell|Golgi|autophag|mitochondr|vesicle|endoplasmic reticulum|purin|glutamyl|synucl|amylo|proteasome"
  topgo_bio <- ENRICH[["GO"]] %>% 
    as_tibble() %>%
    mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
    dplyr::filter(DE > 1 & p.adjust < 0.05) %>%
    dplyr::filter(grepl(pd_terms, Term)) %>% 
    mutate(enrichment_scores = -log10(P.DE)) %>%
    mutate(gene_ratio = DE /N ) %>%
    group_by(Ont) %>% 
    arrange(Ont, desc(gene_ratio), p.adjust) %>%
    mutate(row_num = row_number()) %>%
    filter(row_num <= 15) %>%
    dplyr::select(-row_num) #%>%
  #  filter(Ont != "MF")
  
  plot_name <- paste0("ENRICH_GO_", gsub('.{4}$', '', f), "_pval.pdf")
  pdf(file = file.path(OUT_DIR_PPI, plot_name), width = 7, height = 7)
  topgo_p %>%
    ggplot(aes(x = gene_ratio, y = reorder(Term, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    facet_grid(Ont ~ ., scales = "free") +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (GO)") + 
    theme_bw()
  dev.off()
  
  plot_name <- paste0("ENRICH_GO_", gsub('.{4}$', '', f), "_generatio.pdf")
  pdf(file = file.path(OUT_DIR_PPI, plot_name), width = 7.5, height = 7)
  topgo_gr %>%
    ggplot(aes(x = gene_ratio, y = reorder(Term, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    facet_grid(Ont ~ ., scales = "free") +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (GO)") + 
    theme_bw()
  dev.off()
  
  plot_name <- paste0("ENRICH_GO_", gsub('.{4}$', '', f), "_bio.pdf")
  pdf(file = file.path(OUT_DIR_PPI, plot_name), width = 9, height = 7)
  topgo_bio %>%
    ggplot(aes(x = gene_ratio, y = reorder(Term, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    facet_grid(Ont ~ ., scales = "free") +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (GO)") +
    theme_bw() +
    theme(axis.text.y = element_text(size = 9, hjust = 1, vjust = 0.5, angle = 0),
          legend.position = "right")
  dev.off()
  

}



# -SSN--------------------------------------------------------------------------
# load all genes ssn
DATA.FILE <- "../graph-analysis/patient-network/data/data_expr_4ML_DIAGNOSIS.tsv"
data <- vroom(DATA.FILE, show_col_types = FALSE)
all_genes <- colnames(data)[-1]
all_genes_entrez <- translate_ensembl2entrez(all_genes, ids2ensembldb)
rm(data)


# load relevant genes in the ppi network
FILES <- c("cv_overlappingfeatures.csv")
for (f in FILES){
  genes <- vroom(file.path(OUT_DIR_SSN, f), col_types = cols(), show_col_types = FALSE)
  if (grepl("overlappingfeatures", f)){
    genes_entrez <- translate_ensembl2entrez(genes$Feature, ids2ensembldb)
  } 
  ENRICH[["KEGG"]] <- gsea_kegg(genes_entrez)
  ENRICH[["GO"]] <- gsea_go(genes_entrez)
  meshParams <- new("MeSHHyperGParams", 
                    geneIds = genes_entrez, 
                    universeGeneIds = unique(all_genes_entrez),
                    annotation = "MeSH.Hsa.eg.db",
                    #  meshdb = "MeSH.db", # new requirement for BioC 3.14+
                    category = "C", 
                    database = "gendoo", 
                    pvalueCutoff = 1,
                    pAdjust = "BH")
  ENRICH[["MeSH"]] <- mesh_enrich(genes_entrez, all_genes_entrez, target_meshterms, meshParams)
  for (i in names(ENRICH)){
    if (!grepl("MeSH", i)) {
      ENRICH[[i]] <- ENRICH[[i]] %>% 
        as_tibble() %>%
        mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
        dplyr::filter(p.adjust < 0.05) %>%
        mutate(enrichment_scores = -log10(p.adjust)) %>%
        mutate(gene_ratio = DE /N ) 
    }
    readr::write_csv(ENRICH[[i]], file = file.path(OUT_DIR_SSN, paste0("enrichment_", i, "_", f)))
    print(knitr::kable(ENRICH[[i]][1:30,], row.names = F))
  }

  # plots
  # ORA - KEGG
  plot_name <- paste0("ENRICH_KEGG_", gsub('.{4}$', '', f), ".pdf")
  pdf(file = file.path(OUT_DIR_SSN, plot_name), width = 7, height = 7)
  ENRICH[["KEGG"]] %>%
    arrange(p.adjust) %>%
    slice_min(n=40, p.adjust) %>% 
    ggplot(aes(x = gene_ratio, y = reorder(Pathway, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (KEGG)") +
    theme_minimal() 
  
  dev.off()
  
  # ORA - GO
  topgo_p <- ENRICH[["GO"]] %>% 
    as_tibble() %>%
    mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
    dplyr::filter(DE > 1 & p.adjust < 0.05 & N >= 3) %>%
    mutate(enrichment_scores = -log10(P.DE)) %>%
    mutate(gene_ratio = DE /N ) %>%
    group_by(Ont) %>% 
    arrange(Ont, p.adjust, desc(gene_ratio)) %>%
    mutate(row_num = row_number()) %>%
    filter(row_num <= 10) %>%
    dplyr::select(-row_num)
  
  topgo_gr <- ENRICH[["GO"]] %>% 
    as_tibble() %>%
    mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
    dplyr::filter(DE > 1 & p.adjust < 0.05 & N >= 3) %>%
    mutate(enrichment_scores = -log10(P.DE)) %>%
    mutate(gene_ratio = DE /N ) %>%
    group_by(Ont) %>% 
    arrange(Ont, desc(gene_ratio), p.adjust) %>%
    mutate(row_num = row_number()) %>%
    filter(row_num <= 20) %>%
    dplyr::select(-row_num) %>%
    filter(Ont != "MF")
  
  pd_terms <- "Parkin|parkin|neuron|ubiquit|dopamine|carnitine|beta-oxidation|bile|T cell|Golgi|autophag|mitochondr|vesicle|endoplasmic reticulum|purin|glutamyl|synucl|amylo|proteasome"
  topgo_bio <- ENRICH[["GO"]] %>% 
    as_tibble() %>%
    mutate(p.adjust = p.adjust(P.DE, method = "fdr")) %>%
    dplyr::filter(DE > 1 & p.adjust < 0.05) %>%
    dplyr::filter(grepl(pd_terms, Term)) %>% 
    mutate(enrichment_scores = -log10(P.DE)) %>%
    mutate(gene_ratio = DE /N ) %>%
    group_by(Ont) %>% 
    arrange(Ont, desc(gene_ratio), p.adjust) %>%
    mutate(row_num = row_number()) %>%
    filter(row_num <= 15) %>%
    dplyr::select(-row_num) #%>%
  #  filter(Ont != "MF")
  
  plot_name <- paste0("ENRICH_GO_", gsub('.{4}$', '', f), "_pval.pdf")
  pdf(file = file.path(OUT_DIR_SSN, plot_name), width = 7, height = 7)
  topgo_p %>%
    ggplot(aes(x = gene_ratio, y = reorder(Term, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    facet_grid(Ont ~ ., scales = "free") +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (GO)") + 
    theme_bw()
  dev.off()
  
  plot_name <- paste0("ENRICH_GO_", gsub('.{4}$', '', f), "_generatio.pdf")
  pdf(file = file.path(OUT_DIR_PPI, plot_name), width = 7.5, height = 7)
  topgo_gr %>%
    ggplot(aes(x = gene_ratio, y = reorder(Term, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    facet_grid(Ont ~ ., scales = "free") +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (GO)") + 
    theme_bw()
  dev.off()
  
  plot_name <- paste0("ENRICH_GO_", gsub('.{4}$', '', f), "_bio.pdf")
  pdf(file = file.path(OUT_DIR_SSN, plot_name), width = 9, height = 7)
  topgo_bio %>%
    ggplot(aes(x = gene_ratio, y = reorder(Term, gene_ratio), size = DE, color = p.adjust)) +
    geom_point() +
    facet_grid(Ont ~ ., scales = "free") +
    scale_color_continuous(low = "red", high = "blue") + #, name = c("P-value")) +
    #scale_size(name = "Count") +
    labs(x="GeneRatio", y=NULL, 
         size="Count", col="adj. P-value", 
         title="Over-representation analysis (GO)") +
    theme_bw() +
    theme(axis.text.y = element_text(size = 9, hjust = 1, vjust = 0.5, angle = 0),
          legend.position = "right")
  dev.off()
  
  
}



# Session info -----------------------------------------------------------------
rm(list = ls())
gc(T)
cat("\n================\n  SESSION INFO\n================\n")
sessionInfo()



