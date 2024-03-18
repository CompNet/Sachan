# Check the allegiances, in particular houses and their hierarchical relations.
# Script used only once, to prepare the data.
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/common/prepare_allegiances.R")
###############################################################################




###############################################################################
# get the list of houses and their relations
house.file <- "in/houses.csv"
houses <- read.csv(house.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)

# get the list of characters and their allegiances
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)

# check allegiance distribution
table(unlist(strsplit(char.tab[,"AllegianceBoth"],", ")))

unique.alg <- sort(unique(unlist(strsplit(char.tab[,"AllegianceBoth"],", "))))
idx <- which(
		startsWith(x=unique.alg,prefix="House ") & 
		!(unique.alg %in% c("House of Kisses","House of Black and White")))
idx2 <- match(unique.alg[idx], houses[,"House"])
cbind(unique.alg[idx],houses[idx2,"House"])
