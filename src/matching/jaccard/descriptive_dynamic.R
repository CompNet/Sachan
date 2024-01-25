# Computes the similarity between a character and its counterpart in a different
# network, considering the evolution through time.
#
# Note about the sim evolution plot for the instant graphs: there are very few
# characters that appear in two consecutive chapters, hence the discontinuities.
# 
# Author: Vincent Labatut
# 08/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/jaccard/descriptive_dynamic.R")
###############################################################################
library("igraph")
library("viridis")
library("plot.matrix")
library("scales")

source("src/common/stats.R")




###############################################################################
# processing parameters
COMMON_CHARS_ONLY <- TRUE	# all named characters, or only those common to both compared graphs
CUMULATIVE <- TRUE			# use the instant or cumulative networks
MEAS <- "jaccard"			# no alternative for now
TOP_CHAR_NBR <- 20			# number of important characters




###############################################################################
# output folder
out.folder <- file.path("out","matching","first_2",MEAS)
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)

{	if(COMMON_CHARS_ONLY)
		mode.folder <- "common"
	else
		mode.folder <- "named"
	
	if(CUMULATIVE)
		file.pref <- "sim_dyn-cum_"
	else
		file.pref <- "sim_dyn-inst_"
}




###############################################################################
# only take the first two narrative units (whole narrative not supported here)
NARRATIVE_PART <- 2
# load the static graphs and rank the characters by importance
source("src/common/load_static_nets.R")
# load the dynamic graphs
source("src/common/load_dynamic_nets.R")




###############################################################################
# start matching
gs <- list(gs.nv, gs.cx)			# gs.tv
g.names <- c("novels","comics")		# "tvshow"

# loop over pairs of networks
for(i in 1:(length(gs)-1))
{	cat("..Processing first network ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second network ",g.names[j],"\n",sep="")
		
		# init local folder
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		local.folder <- file.path(out.folder, mode.folder, comp.name)
		dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
		
		# init perf tables
		cnames <- c(comp.name, paste0(g.names[j],"_vs_",g.names[i]), "overall")
		perf.tab.all <- matrix(NA, nrow=max(length(gs[[i]]),length(gs[[j]])), ncol=length(cnames))
		colnames(perf.tab.all) <- cnames
		perf.tab.top <- matrix(NA, nrow=max(length(gs[[i]]),length(gs[[j]])), ncol=length(cnames))
		colnames(perf.tab.top) <- cnames
#		perf.test.all <- matrix(NA, nrow=max(length(gs[[i]]),length(gs[[j]])), ncol=length(cnames))		# test
#		colnames(perf.test.all) <- cnames																# test
		# init sim diff table
		sim.diff <- matrix(NA, nrow=length(ranked.chars), ncol=max(length(gs[[i]]),length(gs[[j]])))
		rownames(sim.diff) <- ranked.chars
		# init best match tables
		best.matches1 <- matrix(NA, nrow=length(ranked.chars), ncol=max(length(gs[[i]]),length(gs[[j]])))
		rownames(best.matches1) <- ranked.chars
		best.matches2 <- matrix(NA, nrow=length(ranked.chars), ncol=max(length(gs[[i]]),length(gs[[j]])))
		rownames(best.matches2) <- ranked.chars
		
		k <- 1
		while(k<=length(gs[[i]]) && k<=length(gs[[j]]))
		{	cat("........Processing iteration ",k,"\n",sep="")
			g1 <- gs[[i]][[k]]
			g2 <- gs[[j]][[k]]
			
			# focus on characters common to both networks
			if(COMMON_CHARS_ONLY)
			{	# remove chars that are not common
				names <- intersect(V(g1)$name,V(g2)$name)
				idx1 <- which(!(V(g1)$name %in% names))
				g1 <- delete_vertices(g1,idx1)
				idx2 <- which(!(V(g2)$name %in% names))
				g2 <- delete_vertices(g2,idx2)
				# set the same character order in both graphs
				idx1 <- match(V(g1)$name, names)
				idx2 <- match(V(g2)$name, names)
				g1 <- permute(graph=g1, permutation=idx1)
				g2 <- permute(graph=g2, permutation=idx2)
			}
			# or keep all chars, but complete the graphs to have the same characters in both
			else 
			{	# get the name lists
				names0 <- intersect(V(g1)$name,V(g2)$name)
				names1 <- setdiff(V(g1)$name,V(g2)$name)
				names2 <- setdiff(V(g2)$name,V(g1)$name)
				names <- c(names0, names1, names2)
				# complete first graph
				attrs2 <- list()
				for(att in vertex_attr_names(graph=g2))
					attrs2[[att]] <- vertex_attr(graph=g2, name=att, index=match(names2, V(g2)$name))
				g1 <- add_vertices(g1, nv=length(names2), attr=attrs2)
				idx1 <- match(V(g1)$name, names)
				g1 <- permute(graph=g1, permutation=idx1)
				# complete second graph
				attrs1 <- list()
				for(att in vertex_attr_names(graph=g1))
					attrs1[[att]] <- vertex_attr(graph=g1, name=att, index=match(names1, V(g1)$name))
				g2 <- add_vertices(g2, nv=length(names1), attr=attrs1)
				idx2 <- match(V(g2)$name, names)
				g2 <- permute(graph=g2, permutation=idx2)
			}
			
			if(gsize(g1)>0 && gsize(g2)>0)
			{	# compute and normalize adjacency matrices
				a1 <- as_adjacency_matrix(graph=g1, type="both", attr="weight", sparse=FALSE)
				a1 <- t(apply(a1, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
				a2 <- as_adjacency_matrix(graph=g2, type="both", attr="weight", sparse=FALSE)
				a2 <- t(apply(a2, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
				names <- V(g1)$name
				
				if(MEAS=="jaccard")
				{	# compute jaccard (weighted) similarity
					sim.mat <- matrix(NA, nrow=nrow(a1), ncol=nrow(a2))
					rownames(sim.mat) <- names
					colnames(sim.mat) <- names
					for(v1 in 1:nrow(a1))
					{	#print(v1)
						for(v2 in 1:nrow(a2)) 
						{	w.min <- pmin(a1[v1,], a2[v2,])
							w.max <- pmax(a1[v1,], a2[v2,])
							if(sum(w.max)==0)
								sim <- 0
							else
								sim <- sum(w.min)/sum(w.max)
							#print(sim)
							sim.mat[v1,v2] <- sim
						}
					}
				}
				
				ranked.names <- setdiff(ranked.chars, setdiff(ranked.chars, names))
				idx <- match(ranked.names, names)
				
				# identify best matches at each time step
#				best.match0[ranked.names,k] <- sapply(1:nrow(sim.mat), function(k) 
#				{	mx1 <- max(sim.mat[k,])
#					mx2 <- max(sim.mat[,k])
#					if(mx1==0 && mx2==0)
#						res <- NA
#					else if(mx1>mx2)
#						res <- sample(colnames(sim.mat)[sim.mat[k,]==mx1], size=1)
#					else if(mx1<mx2)
#						res <- sample(rownames(sim.mat)[sim.mat[,k]==mx2], size=1)
#					else
#						res <- sample(union(colnames(sim.mat)[sim.mat[k,]==mx1], rownames(sim.mat)[sim.mat[,k]==mx2]), size=1)
#					return(res)
#				})
				best.matches1[ranked.names,k] <- apply(sim.mat[idx,idx], 2, function(col)
				{	mx <- max(col)
					if(mx==0)
						res <- NA
					else 
						res <- sample(names(col)[col==mx], size=1)
					return(res)
				})
				perf.test1 <- length(which(best.matches1[,k]==rownames(best.matches1)))/length(ranked.names)
				best.matches2[ranked.names,k] <- apply(sim.mat[idx,idx], 1, function(row) 
				{	mx <- max(row)
					if(mx==0)
						res <- NA
					else 
						res <- sample(names(row)[row==mx], size=1)
					return(res)
				})
				perf.test2 <- length(which(best.matches2[,k]==rownames(best.matches2)))/length(ranked.names)
				#print(perf.test1);print(perf.test2)
				#perf.test.all[k,] <- c(perf.test1, perf.test2, NA)	# test
				
				# compute some matching performance
				sim.self <- diag(sim.mat)
				tmp <- sim.mat; diag(tmp) <- 0
				sim.alter1 <- apply(tmp, 1, max)
				sim.alter2 <- apply(tmp, 2, max)
				sim.alter <- pmax(sim.alter1, sim.alter2)
				diff <- sim.self - sim.alter
				sim.diff[ranked.names,k] <- diff[idx]
				write.csv(x=sim.diff, file=file.path(local.folder,paste0(file.pref,"simdiff.csv")), row.names=FALSE, fileEncoding="UTF-8")
				d1 <- degree(g1,mode="all")
				d2 <- degree(g1,mode="all")
				
				# perf for all characters
				acc1 <- length(which(sim.self>sim.alter2))/length(d1>0)
				acc2 <- length(which(sim.self>sim.alter1))/length(d2>0)
				acc <- length(which(sim.self>sim.alter))/length(sim.self)
				perf.tab.all[k,] <- c(acc1, acc2, acc)
				write.csv(x=perf.tab.all, file=file.path(local.folder,paste0(file.pref,"perf_all.csv")), row.names=FALSE, fileEncoding="UTF-8")
				
				# perf for only top characters
				idx.top <- idx[1:min(length(idx),TOP_CHAR_NBR)]
				acc1 <- length(which(sim.self[idx.top]>sim.alter2[idx.top]))/length(d1[idx.top]>0)
				acc2 <- length(which(sim.self[idx.top]>sim.alter1[idx.top]))/length(d2[idx.top]>0)
				acc <- length(which(sim.self[idx.top]>sim.alter[idx.top]))/length(sim.self[idx.top])
				perf.tab.top[k,] <- c(acc1, acc2, acc)
				write.csv(x=perf.tab.top, file=file.path(local.folder,paste0(file.pref,"perf_top20.csv")), row.names=FALSE, fileEncoding="UTF-8")
				#cat("Performance when matching to the most similar character:\n",sep="");print(perf.tab.all[k,]);print(perf.tab.top)
			}
			
			k <- k + 1
		}
	
		# plot proportions of correct matches
		colors <- brewer_pal(type="qual", palette=2)(ncol(perf.tab.all))
		xs <- 1:nrow(perf.tab.all)
		plot.file <- file.path(local.folder,paste0(file.pref,"perf_all"))
		pdf(paste0(plot.file,".pdf"), bg="white")
			plot(
				NULL,
				xlim=range(xs), ylim=c(0,1),
				xlab="Time", ylab="Proportion of correct matches"
			)
			for(k in 1:ncol(perf.tab.all))
				lines(x=xs, y=perf.tab.all[,k], col=colors[k])
				#lines(x=xs, y=perf.test.all[,1], col="BLACK", lty=3)	# test		
				#lines(x=xs, y=perf.test.all[,2], col="GRAY", lty=3)	# test
			legend(
				x="bottomleft",
				legend=colnames(perf.tab.all),
				fill=colors
			)
		dev.off()
		# focus on top 20 characters
		plot.file <- file.path(local.folder,paste0(file.pref,"perf_top20"))
		pdf(paste0(plot.file,".pdf"), bg="white")
			plot(
				NULL,
				xlim=range(xs), ylim=c(0,1),
				xlab="Time", ylab="Proportion of correct matches"
			)
			for(k in 1:ncol(perf.tab.top))
				lines(x=xs, y=perf.tab.top[,k], col=colors[k])
			legend(
					x="bottomleft",
					legend=colnames(perf.tab.top),
					fill=colors
			)
		dev.off()
		
		# similarity difference
		selected.chars <- 1:5
		colors <- brewer_pal(type="qual", palette=2)(length(selected.chars))
		plot.file <- file.path(local.folder,paste0(file.pref,"simdiff"))
		pdf(paste0(plot.file,".pdf"), bg="white")
			plot(
				NULL,
				xlim=range(xs), ylim=c(-1,1),
				xlab="Time", ylab="Similarity difference between self and best alter"
			)
			abline(h=0, col="BLACK", lty=3)
			for(k in setdiff(1:nrow(sim.diff),selected.chars))
			{	if(CUMULATIVE)
					lines(x=xs, y=sim.diff[k,], col=adjustcolor("GRAY",alpha.f=0.3), lwd=2)
				else
					points(x=xs, y=sim.diff[k,], col=adjustcolor("GRAY",alpha.f=0.3), pch=16)
			}
			for(k in 1:length(selected.chars))
			{	if(CUMULATIVE)
					lines(x=xs, y=sim.diff[selected.chars[k],], col=colors[k], lwd=2)
				else
					points(x=xs, y=sim.diff[selected.chars[k],], col=colors[k], pch=16)
			}
			legend(x="bottomleft", legend=ranked.chars[selected.chars], fill=colors, bg="WHITE")
		dev.off()
			
		# compute performance over whole time series (g1 vs g2)
		best.matches1 <- best.matches1[apply(best.matches1,1,function(row) !all(is.na(row))),]
		dyn.matches1 <- apply(best.matches1, 1, function(row) mode(row,na.rm=TRUE))
		tab1 <- data.frame(rownames(best.matches1),dyn.matches1)
		colnames(tab1) <- c("Character","Match")
		write.csv(x=tab1, file=file.path(local.folder,paste0(file.pref,"series_",comp.name,"_all.csv")), row.names=FALSE, fileEncoding="UTF-8")
		corr.all1 <- tab1[tab1[,1]==tab1[,2],1]
		perf.all1 <- length(corr.all1)/nrow(tab1)
		corr.top1 <- tab1[1:TOP_CHAR_NBR,1][tab1[1:TOP_CHAR_NBR,1]==tab1[1:TOP_CHAR_NBR,2]]
		perf.top1 <- length(corr.top1)/TOP_CHAR_NBR
		perf.tab <- c(perf.all1,perf.top1)
		
		# same, in the other direction (g2 vs g1)
		best.matches2 <- best.matches2[apply(best.matches2,1,function(row) !all(is.na(row))),]
		dyn.matches2 <- apply(best.matches2, 1, function(row) mode(row,na.rm=TRUE))
		tab2 <- data.frame(rownames(best.matches2),dyn.matches2)
		colnames(tab2) <- c("Character","Match")
		write.csv(x=tab2, file=file.path(local.folder,paste0(file.pref,"series_",g.names[j],"_vs_",g.names[i],"_all.csv")), row.names=FALSE, fileEncoding="UTF-8")
		corr.all2 <- tab2[tab2[,1]==tab2[,2],1]
		perf.all2 <- length(corr.all2)/nrow(tab2)
		corr.top2 <- tab2[1:TOP_CHAR_NBR,1][tab2[1:TOP_CHAR_NBR,1]==tab2[1:TOP_CHAR_NBR,2]]
		perf.top2 <- length(corr.top2)/TOP_CHAR_NBR
		perf.tab <- cbind(perf.tab, c(perf.all2,perf.top2))
		
		# overall performance (both directions at once)
		corr.all <- intersect(corr.all1, corr.all2)
		perf.all <- length(corr.all)/length(union(tab1[,1],tab2[,1]))
		cat("  Number of characters considered (all):",length(union(tab1[,1],tab2[,1])),"\n")
		corr.top <- intersect(corr.top1, corr.top2)
		perf.top <- length(corr.top)/length(union(tab1[1:TOP_CHAR_NBR,1],tab2[1:TOP_CHAR_NBR,1]))
		cat("  Number of characters considered (top-20):",length(union(tab1[1:TOP_CHAR_NBR,1],tab2[1:TOP_CHAR_NBR,1])),"\n")
		perf.tab <- cbind(perf.tab, c(perf.all,perf.top))
		
		# finalize table
		rownames(perf.tab) <- c("All","Top-20")
		colnames(perf.tab) <- c(comp.name, paste0(g.names[j], "_vs_", g.names[i]), "overall")
		# record perfs
		write.csv(x=perf.tab, file=file.path(local.folder,paste0(file.pref,"series_perf.csv")), row.names=FALSE, fileEncoding="UTF-8")
		cat("Performance when matching to the most similar character over the whole series:\n",sep="");print(perf.tab)
		
		# NOTE: not possible to produce the same plots as for the static method,
		#       as we don't have an overall similarity matrix, in this case
	}
}
