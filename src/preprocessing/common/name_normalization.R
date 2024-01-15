# Lists the names used in the different media, in order to normalize them.
# Script used only once, to prepare the name conversion maps.
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/common/name_normalization.R")
###############################################################################
library("igraph")




###############################################################################
# read the static tvshow graph
g.tv <- read.graph("in/tvshow/cumul/episode/cumulative_72.graphml", format="graphml")
names.tv <- sort(V(g.tv)$name)
unnamed.tv <- 201	# counted manually in the Excel file
#tab.file <- "in/tvshow/charlist.csv"
#write.csv(x=names.tv, file=tab.file, row.names=FALSE, fileEncoding="UTF-8")




###############################################################################
# read the static novel graph
g.nv <- read.graph("in/novels/cumul/5.ADwD_71_cumul.graphml", format="graphml")
names.nv <- sort(V(g.nv)$name)
unnamed.nv <- 52	# counted manually in the Excel file
#tab.file <- "in/novels/charlist.csv"
#write.csv(x=names.nv, file=tab.file, row.names=FALSE, fileEncoding="UTF-8")




###############################################################################
# comics already have been normalized
g.cx <- read.graph("in/comics/cumul/chapter/cum_143.graphml", format="graphml")
names.cx <- sort(V(g.cx)$name)
unnamed.nv <- length(which(!V(g.cx)$Named))
#tab.file <- "in/comics/charlist.csv"
#write.csv(x=names.cx, file=tab.file, row.names=FALSE, fileEncoding="UTF-8")




###############################################################################
# compare the name lists
cn <- c("Novels", "TV Show", "Comics")
stats <- matrix(NA,nrow=length(cn),ncol=length(cn))
rownames(stats) <- colnames(stats) <- cn
stats["Novels","Novels"] <- length(names.nv)
stats["Novels","TV Show"] <- length(intersect(names.nv,names.tv))
stats["Novels","Comics"] <- length(intersect(names.nv,names.cx))
stats["TV Show","TV Show"] <- length(names.tv)
stats["TV Show","Comics"] <- length(intersect(names.tv,names.cx))
stats["Comics","Comics"] <- length(names.cx)
print(stats)
cat("Characters present in all three narratives:",length(intersect(names.nv,intersect(names.tv,names.cx))),"\n")


# compare with normalized list of characters
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
vals <- sapply(list(names.nv, names.tv, names.cx), function(l) length(intersect(l,char.tab[,"Name"])))
stats2 <- cbind(diag(stats),vals,vals/diag(stats)*100)
print(stats2)
