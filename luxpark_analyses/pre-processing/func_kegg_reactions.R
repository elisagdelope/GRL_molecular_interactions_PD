# Title: func_kegg_reactions.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script contains functions that download compounds and reactions data from kegg API (RbioRXN out of CRAN & not maintained). 
# This script also subset relevant fields of information and exports 2 dataframes (compounds, reactions) with corresponding data in graph-analysis/metab-network/data directory

BiocManager::install("KEGGREST")
library(KEGGREST)
library(plyr)
library(readr)
#BiocManager::install("RbioRXN")
#library("RbioRXN")

# get kegg data by ID
get.kegg.byId <- function(keggId) {
  kegg = data.frame()
  i = 1
  while(i <= length(keggId)) {
    
    cat('processing', keggId[i], '\n')
    query <- keggGet(keggId[i:(i+9)])
    
    for(l in 1:length(query)) {
      
      keggRow = query[[l]]
      
      for(j in names(keggRow)) {
        if(j == 'DBLINKS') {
          for(k in 1:length(keggRow$DBLINKS)) {
            db = unlist(strsplit(keggRow$DBLINKS[k], ': '))[1]
            id = unlist(strsplit(keggRow$DBLINKS[k], ': '))[2]
            keggRow[[db]] = id
          }
        } else if (j == 'PATHWAY') {
          for(k in 1:length(keggRow$PATHWAY)) {
            keggRow$PATHWAY[k] = paste(names(keggRow$PATHWAY[k]), keggRow$PATHWAY[k], sep=': ')
          }
          keggRow$PATHWAY = paste(keggRow$PATHWAY, collapse='///')
        } else if (j == 'REFERENCE') {
          keggRow$REFERENCE = paste(keggRow$REFERENCE[[1]]$REFERENCE, collapse='///')
        } else {
          if(length(keggRow[[j]]) > 1) {
            keggRow[[j]] = paste(keggRow[[j]], collapse='///')
          }
        }
      }
      keggRow[['DBLINKS']] = NULL
      keggRow = as.data.frame(keggRow, stringsAsFactors=FALSE)
      kegg = plyr::rbind.fill(kegg, keggRow)
      kegg[is.na(kegg)] = ''
    }
    i = i + 10 
  }
  return(kegg)
}

# get kegg data for reactions & compounds
get.kegg.all <- function() {
    cmp <- keggList("compound")
    reactionEntry = keggList("reaction")
    
    cmpId = names(cmp)
    cmpId = sub('cpd:', '', cmpId)
    
    reactionEntry = names(reactionEntry)
    reactionEntry = sub('rn:', '', reactionEntry)
    
    keggReaction = get.kegg.byId(reactionEntry)
    keggReaction[is.na(keggReaction)] = ""
    
    keggCompound = get.kegg.byId(cmpId)
    keggCompound[is.na(keggCompound)] = ""
    
    # reference
    referIndex = grep('.+', keggReaction$REFERENCE)
    referId = keggReaction[grep('.+', keggReaction$REFERENCE), 'ENTRY']
    referIdUnique = unique(keggReaction[grep('.+', keggReaction$REFERENCE), 'ENTRY'])
    
    redundantIndex = c()
    for(i in referIdUnique) {
      index = grep(i, referId)
      index = referIndex[index[-1]]
      redundantIndex = c(redundantIndex, index)
    }
    
    if(length(redundantIndex) > 0) {
      keggReaction_unique = keggReaction[-redundantIndex,]
    } else {
      keggReaction_unique = keggReaction
    }
    
    result = list()
    result[['reaction']] = keggReaction_unique
    result[['compound']] = keggCompound
    cat('# of reactions:', nrow(keggReaction_unique), '\n')
    cat('# of compounds:', nrow(keggCompound), '\n')
    return(result)
  }


# Extract relevant information  ------------------------------------------------
# Kegg compound:
#ENTRY	- KEGG ID (C number)
#NAME	- Compound name
#FORMULA	- Molecular formula
#REACTION	- KEGG REACTION entries where the compound particiaptes
#PATHWAY	- KEGG PATHWAY entries where the compound participates
#CAS	- Cross-link to CAS database
#PubChem	- Cross-link to PubChem database
#ChEBI	- Cross-link to ChEBI database
#PDB.CCD	- Cross-link to Chemical Component Dictionary

# Kegg reaction:
#ENTRY	- KEGG ID (R number)
#NAME	- Enzyme name
#ENZYME	- E.C number
#RCLASS	- KEGG RPAIR is a collection of substrate-product pairs (reactant pairs) defined for each reaction in KEGG REACTION
#PATHWAY	- KEGG PATHWAY that this reaction participates

keggAll = get.kegg.all()
kegg_compound <- keggAll$compound[, c("ENTRY","NAME","FORMULA","REACTION","PATHWAY","NETWORK","CAS","PubChem","ChEBI","PDB.CCD")]
kegg_reaction <- keggAll$reaction[, c("ENTRY","NAME","ENZYME","RCLASS","PATHWAY")]



# output results ---------------------------------------------------------------
readr::write_csv(kegg_compound, file = "../graph-analysis/metab-network/data/kegg_compounds.csv")
readr::write_csv(kegg_reaction, file = "../graph-analysis/metab-network/data/kegg_reactions.csv")



# Session info -----------------------------------------------------------------
rm(list = ls())
gc(T)
cat("\n================\n  SESSION INFO\n================\n")
sessionInfo()


