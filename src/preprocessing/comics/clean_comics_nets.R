# Normalizes the comics networks, to make them comparable to those 
# based on the novels and the TV show.
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/comics/clean_comics_nets.R")
###############################################################################
library("igraph")




###############################################################################
#net.folder <- "in/comics/cumul/chapter"
#net.folder <- "in/comics/cumul/scene"
#net.folder <- "in/comics/instant/chapter"
net.folder <- "in/comics/instant/scene"
files <- list.files(path=net.folder, pattern=".+\\.graphml")

# read the networks
gs <- list()
for(file in files)
{	path <- file.path(net.folder, file)
	cat("Loading graph \"",path,"\"\n",sep="")
	
	g <- read.graph(file=path, format="graphml")
	gs <- c(gs, list(g))
}

# read the name conversion map
map.file <- "in/comics/charmap.csv"
cat("Conversion map \"",map.file,"\"\n",sep="")
char.tab <- read.csv(map.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
idx <- which(char.tab[,"NormalizedName"]=="")
char.tab[idx,"NormalizedName"] <- char.tab[idx,"ComicsName"]

# clean each network
for(i in 1:length(gs))
{	cat("Processing graph ",i,"/",length(gs),"\n",sep="")
	g <- gs[[i]]
	
	# normalize the names in the networks
	idx <- match(V(g)$name, char.tab[,"ComicsName"])
	if(is.na(any(idx)))
	{	idx <- which(is.na(idx))
		cat("ERROR: Could not find the following names in the map:")
		print(V(g)$name[idx])
		stop("ERROR")
	}
	
	# fix the weight attribute for edges
	E(g)$weight <- E(g)$Duration
	g <- delete_edge_attr(g, "Duration")
	#g <- delete_edge_attr(g, "Occurrences")
	
	# change vertex attribute names
	V(g)$sex <- V(g)$Sex
	g <- delete_vertex_attr(g, "Sex")
	
	gs[[i]] <- g
}

# write the new graph versions
for(i in 1:length(gs))
{	path <- file.path(net.folder, files[i])
	cat("Recording graph ",i,"/",length(gs)," in \"",path,"\"\n",sep="")
	
	g <- gs[[i]]
	g <- write.graph(g, file=path, format="graphml")
}
