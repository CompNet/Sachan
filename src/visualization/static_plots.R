# Produces plots of the static networks.
# 
# Author: Vincent Labatut
# 03/2024
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/visualization/static_plots.R")
###############################################################################
library("igraph")
library("scales")
library("viridis")

source("src/common/colors.R")




###############################################################################
# processing parameters
NARRATIVE_PART <- 0			# take the whole narrative (0) or only the first two (2) or five (5) narrative units
narr.names <- c("comics"="Comics", "novels"="Novels", "tvshow"="TV Show")




###############################################################################
# output folder
out.folder <- file.path("out","visualization","narratives")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)




###############################################################################
# load character importance table
source("src/common/char_importance.R")
tab.file <- file.path("in",paste0("ranked_importance_S",NARRATIVE_PART,".csv"))
char.importance <- read.csv(file=tab.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE, fileEncoding="UTF-8")
ranked.chars <- char.importance[,"Name"]
imp.moy <- char.importance[,"Mean"]
names(imp.moy) <- char.importance[,"Name"]




###############################################################################
# load the static graphs and rank the characters by importance
files <- c("novels"="in/novels/cumul/5.ADwD_72_cumul.graphml", "comics"="in/comics/cumul/scene/cum_1437.graphml", "tvshow"="in/tvshow/cumul/scene/cumulative_4164.graphml")

# remove isolated characters from the graphs
gs <- list()
for(i in 1:length(files))
{	# read graph
	g <- read.graph(files[i], format="graphml")
	
	# normalize weights (max)
	E(g)$weight <- E(g)$weight / max(E(g)$weight)
	
	# delete isolated characters
	deg <- degree(graph=g, mode="all")
	if(any(deg==0))
		g <- delete_vertices(graph=g, v=which(deg==0))
	
	# add vertex attribute: importance
	V(g)$importance <- char.importance[match(V(g)$name,char.importance[,"Name"]),"Mean"]
	idx <- which(is.na(V(g)$importance))
	V(g)[idx]$importance <- min(V(g)$importance,na.rm=TRUE)
	
	# sort vertices by importance
	idx <- rank(V(g)$importance,ties.method="random")
	g <- permute(graph=g, permutation=idx)
	
	# update graph list
	gs[[i]] <- g
}




###############################################################################
# setup the layout
for(i in 1:length(gs))
{	lay.file <- file.path(out.folder,paste0("layout_",g.names[i],".csv"))
	
	###########
	## this part is done once to initialize the layout
	###########
#	# export the graphs, so that the layout can be fine-tuned using Gephi, then imported back here
#	graph.file <- file.path(out.folder, paste0(g.names[i],".graphml"))
#	cat("Exporting graph in \"",graph.file,"\"\n",sep="")
#	write.graph(graph=gs[[i]], file=graph.file, format="graphml")
#	
#	## read the modified graph, get the layout, record as CSV for later use
#	graph.file <- file.path(out.folder, paste0(g.names[i],".graphml"))
#	g <- read.graph(file=graph.file, format="graphml")
#	layout <- data.frame(Name=V(g)$name, X=V(g)$x, Y=V(g)$y)
#	write.csv(x=layout, file=lay.file, row.names=FALSE, fileEncoding="UTF-8")
#	###########
	
	# read the layout and apply to the network
	layout <- read.csv(file=lay.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
	g <- simplify(gs[[i]],remove.loops=TRUE)
	g <- delete_vertices(graph=g, v=which(!(V(g)$name %in% layout[,"Name"])))
	
	# add coordinate attributes
	V(g)$x <- layout[match(V(g)$name,layout[,"Name"]),"X"]
	V(g)$y <- layout[match(V(g)$name,layout[,"Name"]),"Y"]
	# update graph list
	gs[[i]] <- g
}




###############################################################################
# produce the plot files
v.size.min <- c(0.8, 1.5, 1.5)
v.size.max <- c(6.0, 8.0, 8.0)
e.thick.min <- c(1,  1, 1)
e.thick.max <- c(8, 10, 8)

# loop over narratives
for(i in 1:length(gs))
{	cat("Computing narrative ",g.names[i],"\n",sep="")
	
	graph.file <- file.path(out.folder,paste0("static_",g.names[i],".pdf"))
	g <- gs[[i]]
	
	# vertex size
	vals <- V(g)$importance
	v.sizes <- v.size.min[i] + (vals-min(vals))*(v.size.max[i]-v.size.min[i])/(max(vals)-min(vals))
	
	# vertex color
	v.colors <- rep("#808080", gorder(g))
	idx.ranked <- match(ranked.chars[1:5],V(g)$name)
	v.colors[idx.ranked] <- brewer_pal(type="qual", palette=2)(5)
	
	# edge thickness
	vals <- E(g)$weight
	e.thick <- e.thick.min[i] + (vals-min(vals))*(e.thick.max[i]-e.thick.min[i])/(max(vals)-min(vals))
	
	# edge color
	el <- as_edgelist(graph=g, names=FALSE)
	e.colors <- sapply(1:nrow(el), function(e) combine.colors(v.colors[el[e,1]], v.colors[el[e,2]]))
	e.colors <- sapply(1:nrow(el), function(e) 
				if(el[e,1] %in% idx.ranked || el[e,2] %in% idx.ranked)
					adjustcolor(e.colors[e],alpha.f=0.5)
				else
					adjustcolor(e.colors[e],alpha.f=0.25))
	
	# plot graph
	pdf(graph.file)	# bg="white"
		par(mar=c(0,0,0.8,0), oma=c(0,0,0,0))	# margins Bottom Left Top Right
		plot(
			gs[[i]], 
			main=bquote(bolditalic(.(narr.names[g.names[i]]))),
			vertex.size=v.sizes, 
			vertex.label=NA,
			vertex.color=v.colors,
			vertex.frame.width=0.1,
			edge.width=e.thick,
			edge.color=e.colors
		)
	dev.off()
}
