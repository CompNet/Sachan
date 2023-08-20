# Computes various standard centralities for each character, and cluster them 
# based on these features. The goal is to compare the clusters (considered as
# archetypal characters) from one narrative to the other, and the position of
# specific characters in these clusters (does it change depending on the network?).
# 
# Author: Vincent Labatut
# 08/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/centrality/char_centrality.R")
###############################################################################
library("igraph")
library("viridis")
library("scales")
library("plot.matrix")
library("fmsb")
library("cluster")




###############################################################################
# parameters
CENTR_MEAS <- c("degree", "strength", "closeness", "w_closeness", "betweenness", "w_betweenness", "eigenvector", "w_eigenvector")
short.names <- c("degree"="Deg.", "strength"="Str.", "closeness"="Clos.", "w_closeness"="wClo.", "betweenness"="Betw.", "w_betweenness"="wBetw.", "eigenvector"="Eig.", "w_eigenvector"="wEig")
STANDARDIZE <- TRUE			# whether to standardize (z-score) the centrality scores
COMMON_CHARS_ONLY <- FALSE	# all named characters, or only those common to both compared graphs
WHOLE_NARRATIVE <- FALSE	# only take the first two books, all comics, first two seasons (whole narrative not supported here)




###############################################################################
# output folder
{	if(COMMON_CHARS_ONLY)
		comm.folder <- "common"
	else
		comm.folder <- "named"
	if(WHOLE_NARRATIVE)
		narr.folder <- "everything"
	else
		narr.folder <- "first-two"
	if(STANDARDIZE)
		std.folder <- "standardized"
	else
		std.folder <- "raw"
}

out.folder <- file.path("out", "centrality", comm.folder, narr.folder, std.folder)
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)




###############################################################################
# load the static graphs
source("src/common/load_static_nets.R")
# possibly keep only common characters
if(COMMON_CHARS_ONLY)
{	nm.nv <- V(g.nv)$name
	nm.cx <- V(g.cx)$name
	nm.tv <- V(g.tv)$name
	names <- intersect(nm.nv, intersect(nm.cx, nm.tv))
	g.nv <- delete_vertices(graph=g.nv, v=which(!(nm.nv %in% names)))
	g.cx <- delete_vertices(graph=g.cx, v=which(!(nm.cx %in% names)))
	g.tv <- delete_vertices(graph=g.tv, v=which(!(nm.tv %in% names)))
}




###############################################################################
# compute centralities
gs <- list(g.nv, g.cx, g.tv)
g.names <- c("novels","comics","tvshow")

centr.tabs <- list()

# loop over networks
for(i in 1:length(gs))
{	cat("..Computing centralities in network",g.names[i],"\n")
	g <- gs[[i]]
	
	local.folder <- file.path(out.folder, g.names[i])
	dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
	
	# init stat tabs
	centr.tab <- matrix(NA, nrow=gorder(g), ncol=length(CENTR_MEAS))
	rownames(centr.tab) <- V(g)$name
	colnames(centr.tab) <- CENTR_MEAS
			
	# loop over centrality measures
	for(j in 1:length(CENTR_MEAS))
	{	meas <- CENTR_MEAS[j]
		cat("....Computing centrality measure",meas,"\n")
		
		if(meas=="degree")
			vals <- degree(graph=g, mode="all", normalized=FALSE)
		else if(meas=="strength")
			vals <- strength(graph=g, mode="all", weights=E(g)$weight)
		else if(meas=="closeness")
			vals <- closeness(graph=g, mode="all", weights=rep(1,gsize(g)), normalized=FALSE)
		else if(meas=="w_closeness")
			vals <- closeness(graph=g, mode="all", weights=1-E(g)$weight+min(E(g)$weight), normalized=FALSE)		# need to "reverse" the weights, as the measure considers them as distances 
		else if(meas=="betweenness")
			vals <- betweenness(graph=g, directed=FALSE, weights=rep(1,gsize(g)), normalized=FALSE)
		else if(meas=="w_betweenness")
			vals <- betweenness(graph=g, directed=FALSE, weights=1-E(g)$weight+min(E(g)$weight), normalized=FALSE)	# same here
		else if(meas=="eigenvector")
			vals <- eigen_centrality(graph=g, directed=FALSE, scale=FALSE, weights=rep(1,gsize(g)))$vector
		else if(meas=="w_eigenvector")
			vals <- eigen_centrality(graph=g, directed=FALSE, scale=FALSE, weights=E(g)$weight)$vector
		
		# standardize the values to make them comparable
		if(STANDARDIZE)
			vals <- scale(x=vals, center=TRUE, scale=TRUE)
		
		centr.tab[,meas] <- vals
	}
	
	# record centrality values as a CSV file
	write.csv(x=centr.tab, file=file.path(local.folder,"centrality_values.csv"), row.names=TRUE, fileEncoding="UTF-8")
	centr.tabs[[i]] <- centr.tab
	
	# plot correlation matrix
	for(cm in c("pearson","spearman"))
	{	cor.mat <- cor(x=centr.tab, method=cm)
		plot.file <- file.path(local.folder,paste0("corrmat_",cm))
		pdf(paste0(plot.file,".pdf"), bg="white")
			plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=g.names[i], cex.axis=0.7)
			#plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=g.names[i], cex.axis=0.7, breaks=seq(-1,1,0.1))
		dev.off()
	}
	
	# order the characters by importance (better looking plots)
	chars <- setdiff(ranked.chars, setdiff(ranked.chars,V(g)$name))
	idx <- match(chars, V(g)$name)
	mm <- rbind(apply(centr.tab,2,max),apply(centr.tab,2,min))
	
	# radar plots
	selected.chars <- 1:5
	sel.cols <- brewer_pal(type="qual", palette=2)(length(selected.chars))
	cols <- rep(adjustcolor("BLACK",alpha.f=0.3), nrow(centr.tab))
	cols[selected.chars] <- sel.cols
	anames <- short.names[colnames(centr.tab)]
	plot.file <- file.path(local.folder,paste0("radar_all"))
	pdf(paste0(plot.file,".pdf"), bg="white")
		radarchart(
			as.data.frame(rbind(mm,centr.tab[rev(idx),])),	# values to plot 
			#maxmin=FALSE,									# prevents the lib from expecting the first row to contain certain parameters
			title=g.names[i],								# main title
			#axistype=3, axislabcol="BLACK",				# display axis values
			pty=32,											# don't plot points
			pcol=rev(cols),									# line colors
			plty=1,											# only solid lines
			plwd=2,											# line width
			cglty=3, cglwd=1, cglcol="BLACK",				# axis lines
			vlabels=anames									# clean axis names
		)
		legend(x="bottomleft", legend=chars[selected.chars], fill=sel.cols, inset=-0.10, xpd=TRUE)
	dev.off()
	
	# perform clustering
	dd <- dist(centr.tab)
	dendro <- hclust(d=dd, method="ward.D2") # complete single average ward.D ward.D2
	pp <- matrix(data=c(1,1,1,2,1,3,2,2,2,3,2,3,3,3,3,3,3,3),ncol=2,byrow=TRUE)
	pps <- matrix(data=c(7,7,7/2,7,7/3,7,7,7,7*2/3,7,7*2/3,7,7,7,7,7,7,7),ncol=2,byrow=TRUE)
	sil.scores <- c()
	ks <- 2:9
	for(k in ks)
	{	cat("......Clustersing for k=",k,"\n",sep="")
		clusters <- cutree(dendro, k=k)
		print(table(clusters))
		sil <- silhouette(clusters, dd)
		#plot(sil)
		score <- summary(sil)$avg.width
		cat("........Average silhouette:",score,"\n")
		sil.scores <- c(sil.scores, score) 
		
		plot.file <- file.path(local.folder,paste0("radar_k",k))
		pdf(paste0(plot.file,".pdf"), bg="white", width=pps[k,2], height=pps[k,1])
			parameter <- par(mfrow=pp[k,]) #set up the plotting space
			for(j in 1:k)
			{	ii <- which(clusters[idx]==j)
				tt <- paste0("C",j,": ",length(ii)," (%",round(100*length(ii)/length(clusters)),")")
				par(mar=c(0, 0, 1, 0)+0.2)	# margins Bottom Left Top Right
				parameter <- radarchart(
					as.data.frame(rbind(mm,centr.tab[rev(idx[ii]),])), 
					pty=32, pcol=rev(cols[ii]), plty=1, plwd=2, cglty=3, 
					cglwd=1, cglcol="BLACK", vlabels=anames, title=tt)
			}
		dev.off()
	}
	
	# record silhouette scores
	tab.scores <- data.frame("k"=ks, "Silhouette"=sil.scores)
	write.csv(x=tab.scores, file=file.path(local.folder,"silhouette_scores.csv"), row.names=FALSE, fileEncoding="UTF-8")
}
