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
library("latex2exp")

source("src/common/colors.R")




###############################################################################
# parameters
CENTR_MEAS <- c("degree", "strength", "closeness", "w_closeness", "betweenness", "w_betweenness", "eigenvector", "w_eigenvector")
short.names <- c("degree"="Deg.", "strength"="Str.", "closeness"="Clos.", "w_closeness"="W.Clo.", "betweenness"="Betw.", "w_betweenness"="W.Betw.", "eigenvector"="Eig.", "w_eigenvector"="W.Eig")
STANDARDIZE <- TRUE			# whether to standardize (z-score) the centrality scores
COMMON_CHARS_ONLY <- TRUE	# all named characters, or only those common to both compared graphs
NARRATIVE_PART <- 2			# take the whole narrative (0) or only the first two (2) or five (5) narrative units
TOP_CHAR_NBR <- 20			# number of important characters
ATTR_LIST <- c("Sex")		# vertex attributes to consider when plotting: Named Sex Affiliation
narr.names <- c("comics"="Comics", "novels"="Novels", "tvshow"="TV Show")




###############################################################################
# output folder
{	if(COMMON_CHARS_ONLY)
		comm.folder <- "common"
	else
		comm.folder <- "named"
	if(NARRATIVE_PART==0)
		narr.folder <- "whole_narr"
	else if(NARRATIVE_PART==2)
		narr.folder <- "first_2"
	else if(NARRATIVE_PART==5)
		narr.folder <- "first_5"
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
{	for(i in 1:length(gs))
	{	nm <- V(gs[[i]])$name
		if(i==1)
			common.names <- nm
		else
			common.names <- intersect(common.names, nm)
	}
	for(i in 1:length(gs))
	{	g <- gs[[i]]
		nm <- V(g)$name
		g <- delete_vertices(graph=g, v=which(!(nm %in% common.names)))
		gs[[i]] <- g
	}
}




###############################################################################
# compute centralities
centr.tabs <- list()
radar.data <- list()
radar.cols <- list()

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
		
		# remove NaNs
		vals[is.nan(vals)] <- 0
		
		# standardize the values to make them comparable
		if(STANDARDIZE)
			vals <- scale(x=vals, center=TRUE, scale=TRUE)
		
		centr.tab[,meas] <- vals
	}
	
	# record centrality values as a CSV file
	write.csv(x=centr.tab, file=file.path(local.folder,"centrality_values.csv"), row.names=TRUE, fileEncoding="UTF-8")
	centr.tabs[[i]] <- centr.tab
	
	# order the characters by importance (better looking plots)
	chars <- setdiff(ranked.chars, setdiff(ranked.chars,V(g)$name))
	idx <- match(chars, V(g)$name)
	mm <- rbind(apply(centr.tab,2,max),apply(centr.tab,2,min))
	
	# plot correlation matrix
	for(cm in c("pearson","spearman"))
	{	# all characters
		cor.mat <- cor(x=centr.tab, method=cm)
		rownames(cor.mat) <- short.names[rownames(cor.mat)]
		colnames(cor.mat) <- short.names[colnames(cor.mat)]
		plot.file <- file.path(local.folder,paste0("corrmat_all_",cm))
		pdf(paste0(plot.file,".pdf"), width=8, height=7, bg="white")
			par(mar=c(5, 4, 4-2.5, 2+1.05)+0.1)	# margins Bottom Left Top Right
			plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=narr.names[g.names[i]], cex.axis=1.0, breaks=seq(0.1,1,0.1))
			#plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=g.names[i], cex.axis=0.7, breaks=seq(-1,1,0.1))
		dev.off()
		
		# only most important characters
		cor.mat <- cor(x=centr.tab[idx[1:TOP_CHAR_NBR],], method=cm)
		rownames(cor.mat) <- short.names[rownames(cor.mat)]
		colnames(cor.mat) <- short.names[colnames(cor.mat)]
		plot.file <- file.path(local.folder,paste0("corrmat_top",TOP_CHAR_NBR,"_",cm))
		pdf(paste0(plot.file,".pdf"), width=8, height=7, bg="white")
			par(mar=c(5, 4, 4-2.5, 2+1.05)+0.1)	# margins Bottom Left Top Right
			plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=narr.names[g.names[i]], cex.axis=1.0, breaks=seq(0.1,1,0.1))
			#plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=g.names[i], cex.axis=0.7, breaks=seq(-1,1,0.1))
		dev.off()
	}
	
	# radar plots with main characters
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
			title=narr.names[g.names[i]],					# main title
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
	# keep for later plot
	tmp.list <- list(); tmp.list[["none"]] <- as.data.frame(rbind(mm,centr.tab[rev(idx),]))
	radar.data[[i]] <- tmp.list
	tmp.list <- list(); tmp.list[["none"]] <- rev(cols)
	radar.cols[[i]] <- tmp.list
	
	# radar plots with attributes
	for(att in ATTR_LIST)
	{	vals <- vertex_attr(graph=g, name=tolower(att))
		tab <- t(table(vals))
		print(tab)
		write.csv(x=tab, file=file.path(local.folder,paste0("distrib_att=",att,"_all.csv")), row.names=FALSE, fileEncoding="UTF-8")
		if(att=="Sex")
			sel.cols <- ATT_COLORS_SEX
		else
		{	sel.cols <- brewer_pal(type="qual", palette=2)(length(unique(vals)))
			names(sel.cols) <- sort(unique(vals))
		}
		sel.cols.alpa <- sapply(sel.cols, function(col) adjustcolor(col,alpha.f=0.3))
		cols <- rep(adjustcolor("BLACK",alpha.f=0.4), nrow(centr.tab))
		cols[!is.na(vals)] <- sel.cols.alpa[vals[!is.na(vals)]]
		plot.file <- file.path(local.folder,paste0("radar_all_att=",att))
		pdf(paste0(plot.file,".pdf"), bg="white")
			radarchart(
				as.data.frame(rbind(mm,centr.tab[rev(idx),])),	# values to plot 
				#maxmin=FALSE,									# prevents the lib from expecting the first row to contain certain parameters
				title=narr.names[g.names[i]],					# main title
				#axistype=3, axislabcol="BLACK",				# display axis values
				pty=32,											# don't plot points
				pcol=rev(cols),									# line colors
				plty=1,											# only solid lines
				plwd=2,											# line width
				cglty=3, cglwd=1, cglcol="BLACK",				# axis lines
				vlabels=anames									# clean axis names
			)
			legend(x="bottomleft", legend=names(sel.cols), fill=sel.cols, inset=-0.10, xpd=TRUE, title=att)
		dev.off()
		# keep for later plot
		tmp.list <- radar.data[[i]]; tmp.list[[att]] <- as.data.frame(rbind(mm,centr.tab[rev(idx),]))
		radar.data[[i]] <- tmp.list
		tmp.list <- radar.cols[[i]]; tmp.list[[att]] <- rev(cols)
		radar.cols[[i]] <- tmp.list
	}
	
	# perform clustering
	dd <- dist(centr.tab)
	dendro <- hclust(d=dd, method="ward.D2") # complete single average ward.D ward.D2
	pp <- matrix(data=c(1,1,1,2,1,3,2,2,2,3,2,3,3,3,3,3,3,3),ncol=2,byrow=TRUE)
	pps <- matrix(data=c(7,7,7/2,7,7/3,7,7,7,7*2/3,7,7*2/3,7,7,7,7,7,7,7),ncol=2,byrow=TRUE)
	sil.scores <- c()
	ks <- 2:9
	for(k in ks)
	{	cat("......Clustering for k=",k,"\n",sep="")
		clusters <- cutree(dendro, k=k)
		print(table(clusters))
		sil <- silhouette(clusters, dd)
		#plot(sil)
		score <- summary(sil)$avg.width
		cat("........Average silhouette:",score,"\n")
		sil.scores <- c(sil.scores, score) 
		
		# order clusters by increasing centrality, to improve plot readability
		clust.avg <- sapply(1:k, function(kk) mean(centr.tab[clusters==kk,]))
		perm <- rank(clust.avg)
		clusters <- perm[clusters]
		
		# radar plots with main characters
		selected.chars <- 1:5
		sel.cols <- brewer_pal(type="qual", palette=2)(length(selected.chars))
		cols <- rep(adjustcolor("BLACK",alpha.f=0.3), nrow(centr.tab))
		cols[selected.chars] <- sel.cols
		plot.file <- file.path(local.folder,paste0("radar_k",k))
		pdf(paste0(plot.file,".pdf"), bg="white", width=pps[k,2], height=pps[k,1])
			parameter <- par(mfrow=pp[k,]) # set up the plotting space
			for(j in 1:k)
			{	ii <- which(clusters[idx]==j)
				tt <- TeX(paste0("$C_",j,"$",": ",length(ii)," (%",round(100*length(ii)/length(clusters)),")"))
				par(mar=c(0, 0, 1, 0)+0.2)	# margins Bottom Left Top Right
				parameter <- radarchart(
					as.data.frame(rbind(mm,centr.tab[rev(idx[ii]),])), 
					pty=32, pcol=rev(cols[ii]), plty=1, plwd=1.5, cglty=3, 
					cglwd=1, cglcol="BLACK", vlabels=anames, title=tt)
			}
		dev.off()
		
		# radar plots with attributes
		for(att in ATTR_LIST)
		{	vals <- vertex_attr(graph=g, name=tolower(att))
			tab <- matrix(0,nrow=k,ncol=length(unique(vals)))
			rownames(tab) <- 1:k
			colnames(tab) <- sort(unique(vals))
			if(att=="Sex")
				sel.cols <- ATT_COLORS_SEX
			else
			{	sel.cols <- brewer_pal(type="qual", palette=2)(length(unique(vals)))
				names(sel.cols) <- sort(unique(vals))
			}
			sel.cols.alpha <- sapply(sel.cols, function(col) adjustcolor(col,alpha.f=0.3))
			cols <- rep(adjustcolor("BLACK",alpha.f=0.4), nrow(centr.tab))
			cols[!is.na(vals)] <- sel.cols.alpha[vals[!is.na(vals)]]
			anames <- short.names[colnames(centr.tab)]
			plot.file <- file.path(local.folder,paste0("radar_k",k,"_att=",att))
			pdf(paste0(plot.file,".pdf"), bg="white", width=pps[k,2], height=pps[k,1])
				parameter <- par(mfrow=pp[k,]) # set up the plotting space
				for(j in 1:k)
				{	ii <- which(clusters[idx]==j)
					tt <- table(vals[ii])
					tab[j,names(tt)] <- tt
					tt <- TeX(paste0("$C_",j,"$",": ",length(ii)," (%",round(100*length(ii)/length(clusters)),")"))
					par(mar=c(0, 0, 1, 0)+0.2)	# margins Bottom Left Top Right
					parameter <- radarchart(
						as.data.frame(rbind(mm,centr.tab[rev(idx[ii]),])), 
						pty=32, pcol=rev(cols[ii]), plty=1, plwd=1.5, cglty=3, 
						cglwd=1, cglcol="BLACK", vlabels=anames, title=tt)
				}
			dev.off()

			tab <- rbind(tab, colSums(tab))
			rownames(tab)[nrow(tab)] <- "Total"
			tab <- cbind(tab, rowSums(tab))
			colnames(tab)[ncol(tab)] <- "Total"
			print(tab)
			write.csv(x=tab, file=file.path(local.folder,paste0("distrib_att=",att,"_k",k,".csv")), row.names=TRUE, fileEncoding="UTF-8")
		}
	}
	
	# record silhouette scores
	tab.scores <- data.frame("k"=ks, "Silhouette"=sil.scores)
	write.csv(x=tab.scores, file=file.path(local.folder,"silhouette_scores.csv"), row.names=FALSE, fileEncoding="UTF-8")
}

# additional plot containing all three radar plots at once (for all narratives)
for(att in c("none",ATTR_LIST))
{	# set file name
	plot.file <- file.path(out.folder,"radar_all")
	if(att!="none")
		plot.file <- paste0(plot.file,"_att=",att)
	
	# create file
	pdf(paste0(plot.file,".pdf"), bg="white", width=pps[3,2], height=pps[3,1])
	parameter <- par(mfrow=pp[3,]) # set up the plotting space
	for(i in 1:length(gs))
	{	par(mar=c(0, 0, 1, 0)+0.2)	# margins Bottom Left Top Right
		parameter <- radarchart(
				radar.data[[i]][[att]], 
				pty=32, pcol=radar.cols[[i]][[att]], plty=1, plwd=1.5, cglty=3, 
				cglwd=1, cglcol="BLACK", vlabels=short.names[colnames(centr.tab)], title=narr.names[g.names[i]])
	}
	dev.off()
}




###############################################################################
# plots used in the report
#if(STANDARDIZE && NARRATIVE_PART==2)
#{	for(i in 1:length(gs))
#	{	cat("Processing graph ",g.names[i],"\n")
#		
#		g <- gs[[i]]
#		local.folder <- file.path(out.folder, g.names[i])
#		
#		if(COMMON_CHARS_ONLY)
#			fchar <- "common"
#		else
#			fchar <- "named"
#		
#		# all characters
#		src.file <- file.path(local.folder,"corrmat_all_spearman.pdf")
#		tgt.file <- file.path(out.folder,paste0(fchar,"_S",NARRATIVE_PART,"_corrmat_all_spearman_",g.names[i],".pdf"))
#		file.copy(from=src.file, to=tgt.file, overwrite=TRUE)
#		cat("  Copying file \"",src.file,"\" >> \"",tgt.file,"\"\n")
#		
#		# only top 20 characters
#		src.file <- file.path(local.folder,"corrmat_top20_spearman.pdf")
#		tgt.file <- file.path(out.folder,paste0(fchar,"_S",NARRATIVE_PART,"_corrmat_top20_spearman_",g.names[i],".pdf"))
#		file.copy(from=src.file, to=tgt.file, overwrite=TRUE)
#		cat("..Copying file \"",src.file,"\" >> \"",tgt.file,"\"\n")
#	}
#}




###############################################################################
# distance in the centrality space as a function of character importance
if(COMMON_CHARS_ONLY)
{	for(i in 1:(length(gs)-1))
	{	g1 <- gs[[i]]
		nm1 <- V(g1)$name
		idx1 <- match(nm1, char.importance[,"Name"])
		imp <- char.importance[idx,"Mean"]
		
		for(j in (i+1):length(gs))
		{	idx1 <- match(nm1, rownames(centr.tabs[[i]]))
			idx2 <- match(nm1, rownames(centr.tabs[[j]]))
			
			dists <- sqrt(rowSums((centr.tabs[[i]][idx1,]*centr.tabs[[j]][idx2,])^2))
			
			plot.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_dist-vs-imprt"))
			pdf(paste0(plot.file,".pdf"), width=7, height=7, bg="white")
				par(mar=c(5, 4, 4, 2)+0.1)	# margins Bottom Left Top Right
				plot(imp, dists, log="xy", col="RED", xlab="Importance", ylab="Distance")
			dev.off()
		}
	}
}
