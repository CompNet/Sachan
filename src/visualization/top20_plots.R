# Produces plots of the top-20 characters using a fixed layout, which allows comparing
# visually the narratives (on this character set).
# 
# Author: Vincent Labatut
# 01/2024
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/visualization/top20_plots.R")
###############################################################################
library("igraph")
library("viridis")
library("SDMTools")

source("src/common/colors.R")




###############################################################################
# processing parameters
TOP_CHAR_NBR <- 20			# number of important characters to plot
NARRATIVE_PART <- 5			# take the whole narrative (0) or only the first two (2) or five (5) narrative units
narr.names <- c("comics"="Comics", "novels"="Novels", "tvshow"="TV Show")




###############################################################################
# output folder
out.folder <- file.path("out","visualization")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)




###############################################################################
# load the static graphs and rank the characters by importance
source("src/common/load_static_nets.R")

# list the common characters
common.names <- intersect(V(g.nv)$name, intersect(V(g.cx)$name, V(g.tv)$name))

# rank and keep only the top 20 common characters
ranked.names <- setdiff(ranked.chars, setdiff(ranked.chars, common.names))
top.names <- ranked.names[1:TOP_CHAR_NBR]

# remove the non-important characters from the graphs
for(i in 1:length(gs))
{	g <- gs[[i]]
	nm <- V(g)$name
	# normalize the weights (max)
	E(g)$weight <- E(g)$weight / max(E(g)$weight)
	# delete the non-important characters
	g <- delete_vertices(graph=g, v=which(!(nm %in% top.names)))
	# add vertex attribute: importance
	V(g)$importance <- char.importance[match(V(g)$name,char.importance[,"Name"]),"Mean"]
	# update graph list
	gs[[i]] <- g
}




###############################################################################
# setup the common layout
lay.file <- file.path(out.folder,paste0("layout_S",NARRATIVE_PART,".csv"))

##########
# this part is done once to initialize the layout
##########
## export the graphs, so that the layout can be fine-tuned using Gephi, then imported back here
#for(i in 1:length(gs))
#{	graph.file <- file.path(out.folder, paste0(names(gs)[i],".graphml"))
#	write.graph(graph=gs[[i]], file=graph.file, format="graphml")
#}

## read the modified graph, get the layout, record as CSV for later use
#graph.file <- file.path(out.folder, "novels.graphml")
#g <- read.graph(file=graph.file, format="graphml")
#layout <- data.frame(Name=V(g)$name, X=V(g)$x, Y=V(g)$y)
#write.csv(x=layout, file=lay.file, row.names=FALSE, fileEncoding="UTF-8")
##########

# read layout and apply to networks
layout <- read.csv(file=lay.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
for(i in 1:length(gs))
{	g <- simplify(gs[[i]],remove.loops=TRUE)
	# add coordinate attributes
	V(g)$x <- layout[match(V(g)$name,layout[,"Name"]),"X"]
	V(g)$y <- layout[match(V(g)$name,layout[,"Name"]),"Y"]
	# update graph list
	gs[[i]] <- g
}




###############################################################################
# set color palette
edge.transp <- 65
fine <- 500
pal <- viridis(500)

# produce individual plots
for(i in 1:length(gs))
{	g <- gs[[i]]
	
	# set colors
	vcols <- pal[as.numeric(cut(V(g)$importance,breaks=fine))]
	V(g)$vcols <- vcols
	el <- as_edgelist(graph=g, names=FALSE)
	ecols <- apply(el, 1, function(row) make.color.transparent(combine.colors(col1=vcols[row[1]], col2=vcols[row[2]], transparency=50), edge.transp)) 
	E(g)$ecols <- ecols
	#lcols <- sapply(1:gorder(g), function(v) if(V(g)$importance[v]<0.16) "WHITE" else "BLACK")
	
	# set edge width
	ewidth <- E(g)$weight * 20
	E(g)$ewidth <- ewidth
	
	plot.file <- file.path(out.folder, paste0(names(gs)[i],"_S",NARRATIVE_PART,".pdf"))
	pdf(paste0(plot.file,".pdf"), width=7, height=7, bg="white")
		# adjust margins
		#par(mar=c(5, 4, 4, 2)+0.1)	# margins Bottom Left Top Right
		par(mar=c(0,0,0,0)+0.35)		# margins Bottom Left Top Right
		# plot graph
		plot(g, 
			vertex.size=20, vertex.color=vcols, 
			vertex.label.color="BLACK", vertex.label.font=2, 
			edge.color=ecols, edge.width=ewidth
		)
		title(narr.names[names(gs)[i]], line = -1)
		# add legend
		width <- 0.05; height <- 0.3
		x1 <- -1.14; x2 <- x1 + width
		y2 <- -1.09; y1 <- y2 + height
		leg.loc <- cbind(x=c(x1, x2, x2, x1), y=c(y1, y1, y2, y2))
		legend.gradient(
			pnts=leg.loc,
			cols=pal,
			limits=sprintf("%.2f", range(V(g),na.rm=TRUE)),
			title="Importance", 
			cex=0.8
		)
	dev.off()
	
	gs[[i]] <- g
}

# one plot with all graphs and legend
plot.file <- file.path(out.folder, paste0("all_S",NARRATIVE_PART,".pdf"))
pdf(paste0(plot.file,".pdf"), width=21, height=7, bg="white")
	par(mfrow=c(1,3))
	# add graphs
	for(i in 1:length(gs))
	{	g <- gs[[i]]
		# adjust margins
		par(mar=c(0,0,0,0)+0.35)		# margins Bottom Left Top Right
		# plot graph
		plot(g, 
			vertex.size=20, vertex.color=V(g)$vcols, 
			vertex.label.color="BLACK", vertex.label.font=2, 
			edge.color=E(g)$ecols, edge.width=E(g)$ewidth
		)
		title(narr.names[names(gs)[i]], line = -1)
		# add legend
		width <- 0.1; height <- 0.5
		x1 <- -1.11; x2 <- x1 + width
		y2 <- -1.09; y1 <- y2 + height
		leg.loc <- cbind(x=c(x1, x2, x2, x1), y=c(y1, y1, y2, y2))
		legend.gradient(
			pnts=leg.loc,
			cols=pal,
			limits=sprintf("%.2f", range(V(g),na.rm=TRUE)),
			title="Importance", 
			cex=1
		)
	}
dev.off()
