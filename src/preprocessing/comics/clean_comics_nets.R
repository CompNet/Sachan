# Normalizes the comics networks, to make them comparable to those 
# based on the novels and the TV show.
#
# Note: not sure whether I used the durations or occurrences as weight, at first.
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/comics/clean_comics_nets.R")
###############################################################################
library("igraph")




###############################################################################
net.folder <- "in/comics/cumul/chapter"
#net.folder <- "in/comics/cumul/scene"
#net.folder <- "in/comics/instant/chapter"
#net.folder <- "in/comics/instant/scene"

#net.folder <- "D:/Users/Vincent/eclipse/workspaces/Networks/NaNet/data/ASOIAF/networks/scenes/implicit/unfiltered/cumulative/publication/chapter"
#net.folder <- "D:/Users/Vincent/eclipse/workspaces/Networks/NaNet/data/ASOIAF/networks/scenes/implicit/unfiltered/cumulative/publication/scene"
#net.folder <- "D:/Users/Vincent/eclipse/workspaces/Networks/NaNet/data/ASOIAF/networks/scenes/implicit/unfiltered/instant/publication/chapter"
#net.folder <- "D:/Users/Vincent/eclipse/workspaces/Networks/NaNet/data/ASOIAF/networks/scenes/implicit/unfiltered/instant/publication/scene"

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
	
	# check multiple occurrences of the same name
	tt <- table(V(g)$name)
	if(any(tt>1))
	{	cat("ERROR: The same name is used by several vertices:")
		print(which(tt>1))
		stop("ERROR")
	}
	
	if(gorder(g)>0)
	{	# normalize the names in the networks
		idx <- match(V(g)$name, char.tab[,"ComicsName"])
		if(any(is.na(idx)))
		{	idx <- which(is.na(idx))
			cat("ERROR: Could not find the following names in the map:")
			print(V(g)$name[idx])
			stop("ERROR")
		}
		else
			V(g)$name <- char.tab[idx,"NormalizedName"]
		
		# change vertex attribute names
		V(g)$sex <- V(g)$Sex
		g <- delete_vertex_attr(g, "Sex")
#V(g)$named <- V(g)$Named
#g <- delete_vertex_attr(g, "Named")
	}
	# empty graph (no vertices)
	else
	{	# must create a dummy vertex to change the vertex attributes
		g <- add_vertices(g, 1)
		g <- set_vertex_attr(g, name="sex", value=TRUE)
		g <- delete_vertex_attr(g, "Sex")
		g <- delete_vertices(g, 1)
	}
	
	# fix the weight attribute for edges
	if(gsize(g)>0)
	{	E(g)$weight <- E(g)$Occurrences
		g <- delete_edge_attr(g, "Occurrences")
#		g <- delete_edge_attr(g, "Duration")
	}
	# empty graph (no edge)
	else
	{	# must create two dummy vertices and a dummy edge before changing the edge attributes
		g <- add_vertices(g, 2)
		g <- add_edges(g, edges=c(gorder(g),gorder(g)-1))
		g <- set_edge_attr(g, name="weight", value=1)
		g <- delete_edge_attr(g, "Occurrences")
#		g <- delete_edge_attr(g, "Duration")
		g <- delete_vertices(g, c(gorder(g),gorder(g)-1))
	}
	
	gs[[i]] <- g
}

# write the new graph versions
for(i in 1:length(gs))
{	path <- file.path(net.folder, files[i])
	cat("Recording graph ",i,"/",length(gs)," in \"",path,"\"\n",sep="")
	
	g <- gs[[i]]
	g <- write.graph(g, file=path, format="graphml")
}
