# Computes various standard centralities for each character, and cluster them 
# based on these features. The goal is to compare the clusters (considered as
# archetypal characters) from one narrative to the other, and the position of
# specific characters in these clusters (does it change depending on the network?).
# 
# Author: Vincent Labatut
# 08/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/descript/char_centrality.R")
###############################################################################
library("igraph")
library("viridis")
library("scales")
library("plot.matrix")
library("fmsb")
library("cluster")
library("latex2exp")

source("src/common/colors.R")
source("src/common/topo_measures.R")




###############################################################################
# parameters
CENTR_MEAS <- c("degree", "strength", "closeness", "w_closeness", "betweenness", "w_betweenness", "eigenvector", "w_eigenvector")
STANDARDIZE <- TRUE				# whether to standardize (z-score) the centrality scores
CHARSET <- "common"				# all named characters (named), or only those common to both compared graphs (common), or the 20 most important (top)
TOP_CHAR_NBR <- 20				# number of important characters
NARRATIVE_PART <- 2				# take the whole narrative (0) or only the first two (2) or five (5) narrative units




###############################################################################
# plot parameter
ATTR_LIST <- c("Sex")			# vertex attributes to consider when plotting: named Sex Affiliation

# output folder
{	if(CHARSET=="top")
		comm.folder <- paste0(CHARSET,TOP_CHAR_NBR)
	else
		comm.folder <- CHARSET
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
{	# possibly keep only common characters
	if(CHARSET=="common")
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
	# or possibly keep only top characters
	else if(CHARSET=="top")
	{	for(i in 1:length(gs))
		{	g <- gs[[i]]
			nm <- V(g)$name
			g <- delete_vertices(graph=g, v=which(!(nm %in% ranked.chars[1:TOP_CHAR_NBR])))
			gs[[i]] <- g
		}
	}
}




###############################################################################
# compute centralities
centr.tabs <- list()
radar.data <- list()
radar.cols <- list()
best.clusters <- list()

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
		#print(TOPO_MEAS_ALL[[meas]]$foo)
		vals <- TOPO_MEAS_ALL[[meas]]$foo(g)
		
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
	
	# order the characters by importance (better looking radar plots)
	chars <- setdiff(ranked.chars, setdiff(ranked.chars,V(g)$name))
	idx <- match(chars, V(g)$name)
	mm <- rbind(apply(centr.tab,2,max),apply(centr.tab,2,min))
	
	# plot correlation matrix
	for(cm in c("pearson","spearman"))
	{	# all characters
		cor.mat <- cor(x=centr.tab, method=cm)
		rownames(cor.mat) <- TOPO_MEAS_SHORT_NAMES[rownames(cor.mat)]
		colnames(cor.mat) <- TOPO_MEAS_SHORT_NAMES[colnames(cor.mat)]
#dsd <- which(cor.mat<0,arr.ind=TRUE)
#if(nrow(dsd)>0) for(r in 1:nrow(dsd)) cor.mat[dsd[r,1],dsd[r,2]] <- 0
	
		plot.file <- file.path(local.folder,paste0("corrmat_all_",cm))
		pdf(paste0(plot.file,".pdf"), width=8, height=7)	# bg="white"
			par(mar=c(5, 4, 4-2.5, 2+1.05)+0.1)	# margins Bottom Left Top Right
			plot(
				cor.mat, 
				border=NA, col=viridis, 
				las=2, xlab=NA, ylab=NA, main=bquote(bolditalic(.(narr.names[g.names[i]]))), 
				cex.axis=1.0, breaks=seq(0.0,1,0.1)
			)
			#plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=g.names[i], cex.axis=0.7, breaks=seq(-1,1,0.1))
		dev.off()
		
		# only most important characters
		if(CHARSET!="top")
		{	cor.mat <- cor(x=centr.tab[idx[1:TOP_CHAR_NBR],], method=cm)
			rownames(cor.mat) <- TOPO_MEAS_SHORT_NAMES[rownames(cor.mat)]
			colnames(cor.mat) <- TOPO_MEAS_SHORT_NAMES[colnames(cor.mat)]
			plot.file <- file.path(local.folder,paste0("corrmat_top",TOP_CHAR_NBR,"_",cm))
			pdf(paste0(plot.file,".pdf"), width=8, height=7)	# bg="white"
				par(mar=c(5, 4, 4-2.5, 2+1.05)+0.1)	# margins Bottom Left Top Right
				plot(
					cor.mat, 
					border=NA, col=viridis, 
					las=2, xlab=NA, ylab=NA, main=bquote(bolditalic(.(narr.names[g.names[i]]))), 
					cex.axis=1.0, breaks=seq(0.0,1,0.1)
				)
				#plot(cor.mat, border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=g.names[i], cex.axis=0.7, breaks=seq(-1,1,0.1))
			dev.off()
		}
	}
	
	# radar plots with main characters
	selected.chars <- 1:5
	sel.cols <- brewer_pal(type="qual", palette=2)(length(selected.chars))
	cols <- rep(adjustcolor("BLACK",alpha.f=0.3), nrow(centr.tab))
	cols[selected.chars] <- sel.cols
	anames <- TOPO_MEAS_SHORT_NAMES[colnames(centr.tab)]
	plot.file <- file.path(local.folder,paste0("radar_all"))
	pdf(paste0(plot.file,".pdf"))	# bg="white"
		radarchart(
			as.data.frame(rbind(mm,centr.tab[rev(idx),])),			# values to plot 
			#maxmin=FALSE,											# prevents the lib from expecting the first row to contain certain parameters
			title=bquote(bolditalic(.(narr.names[g.names[i]]))),	# main title
			#axistype=3, axislabcol="BLACK",						# display axis values
			pty=32,													# don't plot points
			pcol=rev(cols),											# line colors
			plty=1,													# only solid lines
			plwd=2,													# line width
			cglty=3, cglwd=1, cglcol="BLACK",						# axis lines
			vlabels=anames											# clean axis names
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
		pdf(paste0(plot.file,".pdf"))	# bg="white"
			radarchart(
				as.data.frame(rbind(mm,centr.tab[rev(idx),])),			# values to plot 
				#maxmin=FALSE,											# prevents the lib from expecting the first row to contain certain parameters
				title=bquote(bolditalic(.(narr.names[g.names[i]]))),	# main title
				#axistype=3, axislabcol="BLACK",						# display axis values
				pty=32,													# don't plot points
				pcol=rev(cols),											# line colors
				plty=1,													# only solid lines
				plwd=2,													# line width
				cglty=3, cglwd=1, cglcol="BLACK",						# axis lines
				vlabels=anames											# clean axis names
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
	best.sil <- -1
	for(k in ks)
	{	cat("......Clustering for k=",k,"\n",sep="")
		
		# handle clusters
		clusters <- cutree(dendro, k=k)
		print(table(clusters))
		tab.file <- file.path(local.folder,paste0("membership_k",k,".csv"))
		write.csv(x=clusters, file=tab.file, row.names=TRUE, fileEncoding="UTF-8")
		
		# compute silhouette
		sil <- silhouette(clusters, dd)
		#plot(sil)
		score <- summary(sil)$avg.width
		cat("........Average silhouette:",score,"\n")
		sil.scores <- c(sil.scores, score)
		if(score>best.sil)
		{	best.clusters[[i]] <- clusters
			best.sil <- score
		}
		
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
		pdf(paste0(plot.file,".pdf"), width=pps[k,2], height=pps[k,1])	# bg="white"
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
			anames <- TOPO_MEAS_SHORT_NAMES[colnames(centr.tab)]
			plot.file <- file.path(local.folder,paste0("radar_k",k,"_att=",att))
			pdf(paste0(plot.file,".pdf"), width=pps[k,2], height=pps[k,1])	# bg="white"
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
	pdf(paste0(plot.file,".pdf"), width=pps[3,2], height=pps[3,1])	# bg="white"
	parameter <- par(mfrow=pp[3,]) # set up the plotting space
	for(i in 1:length(gs))
	{	par(mar=c(0, 0, 1, 0)+0.2)	# margins Bottom Left Top Right
		parameter <- radarchart(
				radar.data[[i]][[att]], 
				pty=32, pcol=radar.cols[[i]][[att]], plty=1, plwd=1.5, cglty=3, 
				cglwd=1, cglcol="BLACK", vlabels=TOPO_MEAS_SHORT_NAMES[colnames(centr.tab)], title=bquote(bolditalic(.(narr.names[g.names[i]]))))
	}
	dev.off()
}




###############################################################################
# plots used in the report
gen.folder <- file.path("out", "centrality")
if(STANDARDIZE)
{	for(i in 1:length(gs))
	{	if(NARRATIVE_PART>0 || g.names[i]=="tvshow")
		{	cat("Processing graph ",g.names[i],"\n")
			
			narr.part <- NARRATIVE_PART
			if(NARRATIVE_PART==0)
				narr.part <- 8
			
			g <- gs[[i]]
			local.folder <- file.path(out.folder, g.names[i])
			
			src.file <- file.path(local.folder,"corrmat_all_spearman.pdf")
			tgt.file <- file.path(gen.folder,paste0(comm.folder,"_S",narr.part,"_corrmat_spearman_",g.names[i],".pdf"))
			file.copy(from=src.file, to=tgt.file, overwrite=TRUE)
			cat("  Copying file \"",src.file,"\" >> \"",tgt.file,"\"\n")
		}
	}
}




###############################################################################
# distance in the centrality space as a function of character importance
if(CHARSET=="common")
{	for(i in 1:(length(gs)-1))
	{	g1 <- gs[[i]]
		nm1 <- V(g1)$name
		idx1 <- match(nm1, char.importance[,"Name"])
		imp <- char.importance[idx1,"Mean"]
		
		for(j in (i+1):length(gs))
		{	g2 <- gs[[j]]
			comp.name <- paste0(g.names[i], "_vs_", g.names[j])
			comp.title <- bquote(bolditalic(.(narr.names[g.names[i]]))~bold(" vs. ")~bolditalic(.(narr.names[g.names[j]])))
			
			idx1 <- match(nm1, rownames(centr.tabs[[i]]))
			idx2 <- match(nm1, rownames(centr.tabs[[j]]))
			
			for(meas in c("euclidean","cosine"))
			{	# compute only the distance betwee each character and itself in the other network
				#dists <- sqrt(rowSums((centr.tabs[[i]][idx1,]*centr.tabs[[j]][idx2,])^2))
				
				# compute the distance between all pairs of characters
				dist.mat <- matrix(NA, nrow=length(nm1), ncol=length(nm1))
				rownames(dist.mat) <- nm1
				colnames(dist.mat) <- nm1
				for(v1 in 1:length(nm1))
				{	vect1 <- centr.tabs[[i]][idx1[v1],]
					for(v2 in 1:length(nm1)) 
					{	vect2 <- centr.tabs[[j]][idx2[v2],]
						if(meas=="euclidean")
							dist.mat[v1,v2] <- sqrt(sum((vect1-vect2)^2))
						else
							dist.mat[v1,v2] <- sum(vect1/sqrt(sum(vect1^2)) * vect2/sqrt(sum(vect2^2)))
					}
				}
				
				# plot distance matrix
				ranked.names <- setdiff(ranked.chars, setdiff(ranked.chars, nm1))
				idx <- match(ranked.names, nm1)
				plot.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_dist_matrix_",meas,"_all"))
				pdf(paste0(plot.file,".pdf"), width=7, height=7)	# bg="white"
					par(mar=c(3,2,2,0.5)+0.1)	# margins Bottom Left Top Right
					plot(
						dist.mat[idx,idx], 
						border=NA, col=viridis, 
						las=2, 
						xlab=bquote(italic(.(narr.names[g.names[i]]))), ylab=bquote(italic(.(narr.names[g.names[j]]))), main=NA, 
						axis.col=NULL, axis.row=NULL, mgp=c(1,1,0),
#						key=NULL
					)
					title(comp.title,  line=0.5)
				dev.off()
				# plot only top characters
				plot.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_dist_matrix_",meas,"_top", TOP_CHAR_NBR))
				pdf(paste0(plot.file,".pdf"))	# bg="white"
					par(mar=c(5.5,4.75,4.5,2)+0.1)	# margins Bottom Left Top Right
					plot(dist.mat[idx[1:TOP_CHAR_NBR],idx[1:TOP_CHAR_NBR]], border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=comp.title, cex.axis=0.5, fmt.key="%.2f")
				dev.off()
				
				# compute some sort of performance by considering the most similar alters vs. self
				dist.self <- diag(dist.mat)
				tmp <- dist.mat; diag(tmp) <- 0
				d1 <- degree(g1,mode="all")
				d2 <- degree(g2,mode="all")
				if(meas=="euclidean")
				{	dist.alter1 <- apply(tmp, 1, min)
					dist.alter2 <- apply(tmp, 2, min)
					dist.alter <- pmin(dist.alter1, dist.alter2)
					acc1 <- length(which(dist.self<dist.alter2))/length(d1>0)
					acc2 <- length(which(dist.self<dist.alter1))/length(d2>0)
					acc <- length(which(dist.self<dist.alter))/length(dist.self)
				}
				else
				{	dist.alter1 <- apply(tmp, 1, max)
					dist.alter2 <- apply(tmp, 2, max)
					dist.alter <- pmax(dist.alter1, dist.alter2)
					acc1 <- length(which(dist.self>dist.alter2))/length(d1>0)
					acc2 <- length(which(dist.self>dist.alter1))/length(d2>0)
					acc <- length(which(dist.self>dist.alter))/length(dist.self)
				}
				cat("  Number of characters used to compute the perf:",length(dist.self),"\n")
				perf.tab <- c(acc1,acc2,acc)
				# focus only on most important characters
				idx <- idx[1:TOP_CHAR_NBR]
				if(meas=="euclidean")
				{	acc1 <- length(which(dist.self[idx]<dist.alter2[idx]))/length(d1[idx]>0)
					acc2 <- length(which(dist.self[idx]<dist.alter1[idx]))/length(d2[idx]>0)
					acc <- length(which(dist.self[idx]<dist.alter[idx]))/length(dist.self[idx])
				}
				else
				{	acc1 <- length(which(dist.self[idx]>dist.alter2[idx]))/length(d1[idx]>0)
					acc2 <- length(which(dist.self[idx]>dist.alter1[idx]))/length(d2[idx]>0)
					acc <- length(which(dist.self[idx]>dist.alter[idx]))/length(dist.self[idx])
				}
				perf.tab <- rbind(perf.tab, c(acc1,acc2,acc))
				cat("  Number of characters used to compute the top-20 perf:",length(idx),"\n")
				rownames(perf.tab) <- c("All","Top-20")
				colnames(perf.tab) <- c(comp.name,paste0(g.names[j], "_vs_", g.names[i]),"overall")
				tab.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_dist_perf_",meas,".csv"))
				write.csv(x=perf.tab, file=tab.file, row.names=TRUE, fileEncoding="UTF-8")
				cat("Performance when matching to the most similar character:\n",sep="");print(perf.tab)
				
				# define a table to store correlation values
				rn <- c("Self-dist_vs_Imprt","Dist-diff_vs_Imprt")
				cn <- c("PearsonCoef","PearsonPval","SpearmanCoef","SpearmanPval","KendallCoef","KendallPval")
				corr.tab <- matrix(NA,nrow=length(rn), ncol=length(cn))
				colnames(corr.tab) <- cn
				rownames(corr.tab) <- rn
				
				# set up colors for next plots	
				transp <- 25	# transparency level
				pal <- get.palette(2)
				cols <- rep(make.color.transparent(pal[2],transp), gorder(g1))
				cols[which(nm1 %in% ranked.chars[1:TOP_CHAR_NBR])] <- make.color.transparent(pal[1],transp)
				
				# plot self-distance vs. character importance
				yvals <- dist.self
				res <- cor.test(x=imp, y=yvals, method="pearson")
				corr.tab["Self-dist_vs_Imprt","PearsonCoef"] <- res$estimate
				corr.tab["Self-dist_vs_Imprt","PearsonPval"] <- res$p.value
				res <- cor.test(x=imp, y=yvals, method="spearman")
				corr.tab["Self-dist_vs_Imprt","SpearmanCoef"] <- res$estimate
				corr.tab["Self-dist_vs_Imprt","SpearmanPval"] <- res$p.value
				res <- cor.test(x=imp, y=yvals, method="kendall")
				corr.tab["Self-dist_vs_Imprt","KendallCoef"] <- res$estimate
				corr.tab["Self-dist_vs_Imprt","KendallPval"] <- res$p.value
				plot.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_dist-vs-imprt_",meas))
				pdf(paste0(plot.file,".pdf"), width=7, height=7)	# bg="white"
					par(mar=c(5, 4, 4, 2)+0.1)	# margins Bottom Left Top Right
					plot(
						NULL,
						log=if(meas=="euclidean") "xy" else "x", 
						main=comp.title, xlab="Importance", ylab=if(meas=="eucliden") "Self-Distance" else "Self-Similarity",
						xlim=range(imp[imp>0]), ylim=range(yvals[yvals>0])
					)
					points(
						imp, yvals,
						pch=16, col=cols, 
					)
					legend(
						x="topleft",
						title="Characters",
						legend=c(paste0("Top-",TOP_CHAR_NBR),"Others"),
						fill=pal
					)
				dev.off()
				
				# plot self-distance-best alter vs. character importance
				yvals <- dist.self-dist.alter
				res <- cor.test(x=imp, y=yvals, method="pearson")
				corr.tab["Dist-diff_vs_Imprt","PearsonCoef"] <- res$estimate
				corr.tab["Dist-diff_vs_Imprt","PearsonPval"] <- res$p.value
				res <- cor.test(x=imp, y=yvals, method="spearman")
				corr.tab["Dist-diff_vs_Imprt","SpearmanCoef"] <- res$estimate
				corr.tab["Dist-diff_vs_Imprt","SpearmanPval"] <- res$p.value
				res <- cor.test(x=imp, y=yvals, method="kendall")
				corr.tab["Dist-diff_vs_Imprt","KendallCoef"] <- res$estimate
				corr.tab["Dist-diff_vs_Imprt","KendallPval"] <- res$p.value
				plot.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_distance-diff_vs_importance_",meas))
				pdf(paste0(plot.file,".pdf"), width=7, height=7)	# bg="white"
					par(mar=c(5, 4, 4, 2)+0.1)	# margins Bottom Left Top Right
					plot(
						NULL, 
						log=if(meas=="euclidean") "xy" else "x", 
						main=comp.title, xlab="Importance", ylab=paste0("Difference between self and best alter ",if(meas=="euclidean") "distances" else "similarities"),
						xlim=range(imp), ylim=range(yvals)
					)
					abline(h=0, lty=2)
					points(
						imp, yvals,
						pch=16, col=cols, 
					)
					legend(
						x="bottomright",
						title="Characters",
						legend=c(paste0("Top-",TOP_CHAR_NBR),"Others"),
						fill=pal
					)
				dev.off()
				
				# record correlation matrix
				cat("Correlation matrix:\n"); print(corr.tab)
				tab.file <- file.path(out.folder, paste0(g.names[i], "_", g.names[j], "_dist-imprt_corr_",meas,".csv"))
				write.csv(x=corr.tab, file=tab.file, row.names=TRUE, fileEncoding="UTF-8")
				
				# note: correlation test
				# 	h_0: no linear relationship between the two variables
				#	p<alpha => reject the null hypothesis, i.e. there is a relationship
			}
		}
	}
}




################################################################################
# compare the best clusters
if(CHARSET!="named")
{	for(metric in c("nmi","ari"))
	{	# parameter of the function used to compute the metric
		param <- metric
		if(metric=="ari")
			param <- "adjusted.rand"
		
		# init matrix to store results
		comp.clstrs <- matrix(NA, nrow=length(gs), ncol=length(gs))
		rownames(comp.clstrs) <- names(gs)
		colnames(comp.clstrs) <- names(gs)
		
		# loop over pairs of networks
		for(i in 1:(length(gs)-1))
		{	for(j in (i+1):length(gs))
			{	# align the characters
				idx <- match(names(best.clusters[[i]]), names(best.clusters[[j]]))
				
				# compute the score
				score <- compare(comm1=best.clusters[[i]], comm2=best.clusters[[j]][idx], method=param)
				cat("Similarity (",metric,") between ",names(gs)[i]," and ",names(gs)[j],": ",score,"\n",sep="")
				comp.clstrs[names(gs)[i],names(gs)[j]] <- comp.clstrs[names(gs)[j],names(gs)[i]] <- score
			}
		}
		
		print(comp.clstrs)
		tab.file <- file.path(out.folder, paste0("cluster_comparison_",metric,".csv"))
		write.csv(x=comp.clstrs, file=tab.file, row.names=TRUE, fileEncoding="UTF-8")
	}
}
