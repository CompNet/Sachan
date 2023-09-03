# Experiments with vertex matching, using the adaptive hard seed approach.
# It is an iterative method where we take the first few best matches (according
# to some heuristic) and use them as hard seeds in the next iteration. By using
# an increasing number of seeds, the estimation is supposed to get better and better.
# 
# Author: Vincent Labatut
# 06/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/igm/exp_adaptive_hard_seeds.R")
###############################################################################
library("igraph")
library("iGraphMatch")
library("scales")




###############################################################################
# processing parameters
MAX_ITER <- 200				# limit on the number of iterations during matching
ATTR <- "none"				# attribute used: none sex affiliation both
WHOLE_NARRATIVE <- FALSE	# only take the first two books, all comics, first two seasons (whole narrative not supported here)
TOP_CHAR_NBR <- 20			# number of important characters




###############################################################################
# output folder
out.folder <- file.path("out","matching",paste0("attr_",ATTR))
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)
mode.folder <- "common_raw_adaptive_hard"




###############################################################################
# load the static graphs
source("src/common/load_static_nets.R")




###############################################################################
# identify most important characters (according to novels)
top.chars <- V(g.nv)$name[order(degree(g.nv),decreasing=TRUE)][1:TOP_CHAR_NBR]




###############################################################################
# adaptive hard seeding
gs <- list(g.nv, g.cx, g.tv)
g.names <- c("novels","comics","tvshow")
methods <- c("convex", "indefinite", "PATH", "percolation", "Umeyama")	# "IsoRank" requires a vertex similarity matrix

tab.exact.matches <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods))
colnames(tab.exact.matches) <- methods
rownames(tab.exact.matches) <- rep(NA,nrow(tab.exact.matches))
r <- 1
sn <- c(0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 130)	# numbers of seeds

# loop over pairs of networks
for(i in 1:(length(gs)-1))
{	cat("..Processing first network ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second network ",g.names[j],"\n",sep="")
		g1 <- gs[[i]]
		g2 <- gs[[j]]
		
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		rownames(tab.exact.matches)[r] <- comp.name
		
		# focus on characters common to both networks
		names <- intersect(V(g1)$name,V(g2)$name)
		idx1 <- which(!(V(g1)$name %in% names))
		g1 <- delete_vertices(g1,idx1)
		idx2 <- which(!(V(g2)$name %in% names))
		g2 <- delete_vertices(g2,idx2)
		
		# build the vertex similarity matrix
		sim.mat <- NULL
		if(ATTR!="none")
		{	sex.mat <- outer(X=V(g1)$sex, Y=V(g2)$sex, FUN=function(x,y) as.integer(x==y))
			aff.mat <- outer(X=V(g1)$affiliation, Y=V(g2)$affiliation, FUN=function(x,y) as.integer(x==y))
			if(ATTR=="sex")
				sim.mat <- sex.mat
			else if(ATTR=="affiliation")
				sim.mat <- aff.mat
			else if(ATTR=="both")
				sim.mat <- (sex.mat + aff.mat)/2
		}
		
		# loop over matching methods
		for(m in 1:length(methods))
		{	method <- methods[m]
			cat("......Applying method ",method,"\n",sep="")
			local.folder <- file.path(out.folder, mode.folder, comp.name, method)
			dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
			
			# loop over seed number
			tab.evol <- c()
			seeds <- NULL
			for(s in sn)
			{	cat("........Number of seeds: ",s,"\n",sep="")
				
				# update seeds
				if(s>0)
				{	bm <- best_matches(A=g1, B=g2, match=res, measure="row_cor")			# "row_cor", "row_diff", or "row_perm_stat"
					bm0 <- bm[!is.na(bm[,"A_best"]) & !is.na(bm[,"B_best"]),]
					seeds_bm <- head(bm0, min(s,nrow(bm0)))
					seeds <- seeds_bm[, 1:2]
				}
				
				if(method=="indefinite")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						similarity=sim.mat,		# vertex-vertex similarity matrix
						
						method="indefinite",	# matching method: indefinite relaxation of the objective function
						start="bari", 			# initialization method for the matrix
						#lap_method=NULL,		# method used to solve LAP
						max_iter=MAX_ITER		# maximum number of replacing matches
					)
				}
				else if(method=="convex")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						similarity=sim.mat,		# vertex-vertex similarity matrix
						
						method="convex",		# matching method: convex relaxation of the objective function
						start="bari", 			# initialization method for the matrix
						#lap_method=NULL,		# method used to solve LAP
						max_iter=MAX_ITER,		# maximum number of replacing matches
						#tol = 1e-05			# tolerance of edge disagreements
					)
				}
				else if(method=="PATH")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						similarity=sim.mat,		# vertex-vertex similarity matrix
						
						method="PATH",			# matching method: ?
						#lap_method=NULL,		# method used to solve LAP
						#epsilon=1,				# small value
						max_iter=MAX_ITER		# maximum number of replacing matches
					)
				}
				else if(method=="percolation")
				{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)	# this method needs at least one seed
					if(s!=0)
						seed <- seeds
					res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seed,				# known vertex matches
						similarity=sim.mat,		# vertex-vertex similarity matrix
						
						method="percolation",	# matching method: percolation
						#r="2",					# threshold of neighboring pair scores
						ExpandWhenStuck=TRUE	# expand the seed set when Percolation algorithm stops before matching all the vertices (better when few seeds)
					)
				}
#				else if(method=="IsoRank")
#				{	res <- gm(
#						A=g1, B=g2,				# graphs to compare 
#						#seeds,					# known vertex matches
#						similarity=,			# vertex-vertex similarity matrix (required for method "IsoRank")
#						
#						method="IsoRank",		# matching method: IsoRank algorithm (spectral method)
#						#lap_method=NULL,		# method used to solve LAP
#						max_iter=MAX_ITER		# maximum number of replacing matches
#					)
#				}
				else if(method=="Umeyama")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						similarity=sim.mat,		# vertex-vertex similarity matrix
						
						method="Umeyama"		# matching method: Umeyama algorithm (spectral)
					)
				}
				
				sink(file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_summary.txt")))
				# ground truth (not useful due to the NAs resulting from different characters in the nets)
				gt <- match(V(g1)$name, V(g2)$name)
				# summary of the matching process
				print(summary(res, g1, g2, true_label=gt))
				# number of perfect matches
				idx.exact.matches <- which(V(g1)$name[res$corr_A]==V(g2)$name[res$corr_B])
				nbr.exact.matches <- length(idx.exact.matches)
				tab.exact.matches[r,method] <- nbr.exact.matches
				tab.evol <- c(tab.evol, nbr.exact.matches)
				cat("Number of perfect matches:",nbr.exact.matches,"\n")
				sink()
				print(summary(res, g1, g2, true_label=gt))
				cat("Number of perfect matches:",nbr.exact.matches,"\n")
				
				# list perfect matches
				if(nbr.exact.matches>0)
				{	exact.matches <- matrix(V(g1)$name[res$corr_A][idx.exact.matches],ncol=1)
					colnames(exact.matches) <- "Character"
					cat("List of exact matches:\n")
					print(exact.matches)
					write.csv(x=exact.matches, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_list_exact_matches.csv")), row.names=FALSE, fileEncoding="UTF-8")
				}
				
				# list all character matches
				tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])
				colnames(tab) <- c("Corr_A","Corr_B")
				#print(tab)
				write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_list_full.csv")), row.names=FALSE, fileEncoding="UTF-8")
				# same thing for the most important characters
				tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])[V(g1)$name[res$corr_A] %in% top.chars | V(g2)$name[res$corr_B] %in% top.chars,]
				colnames(tab) <- c("Top_Corr_A","Top_Corr_B")
				print(tab)
				write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_list_top.csv")), row.names=FALSE, fileEncoding="UTF-8")
				
				# best matches according to some internal criterion (no GT)
				for(meas in c("row_cor", "row_diff", "row_perm_stat"))
				{	cat(".........Computing measure ",meas,"\n",sep="")
					bm <- best_matches(A=g1, B=g2, match=res, measure=meas)			# "row_cor", "row_diff", or "row_perm_stat"
					tab <- cbind(bm,V(g1)$name[bm$A_best], V(g2)$name[bm$B_best])
					colnames(tab)[(ncol(bm)+1):(ncol(bm)+2)] <- c("Character_A","Character_B")
					print(tab[1:TOP_CHAR_NBR,])
					write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_best_matches_",meas,".csv")), row.names=FALSE, fileEncoding="UTF-8")
				}
			}
			
			# record evolution table
			tab.evol <- data.frame(sn, tab.evol)
			colnames(tab.evol) <- c("AdaptiveSeeds","ExactMatches")
			write.csv(x=tab.evol, file=file.path(local.folder,"_exact_matches_evolution.csv"), row.names=FALSE, fileEncoding="UTF-8")
		}
		
		r <- r + 1
	}
}

# record overall table
print(tab.exact.matches)
write.csv(x=tab.exact.matches, file=file.path(out.folder,mode.folder,"exact_matches_comparison.csv"), row.names=TRUE, fileEncoding="UTF-8")




###############################################################################
# plot the evolution of the performance over iterations
#colors <- 1:length(methods)
colors <- brewer_pal(type="qual", palette=2)(length(methods))

for(i in 1:(length(gs)-1))
{	for(j in (i+1):length(gs))
	{	comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		local.folder <- file.path(out.folder, mode.folder, comp.name)
		
		# loop over matching methods
		all.evol <- matrix(NA,nrow=length(sn),ncol=length(methods))
		for(m in 1:length(methods))
		{	# read evolution table
			method <- methods[m]
			tab.evol <- read.csv(file=file.path(local.folder,method,"_exact_matches_evolution.csv"), header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
			all.evol[,m] <- tab.evol[,"ExactMatches"]
		}
		
		# record evolution table
		tab <- data.frame(sn, all.evol)
		colnames(tab) <- c("AdaptiveSeeds", methods)
		write.csv(x=all.evol, file=file.path(local.folder,"exact_matches_evolution.csv"), row.names=TRUE, fileEncoding="UTF-8")
		
		# create plot
		plot.file <- file.path(local.folder,"exact_matches_evolution")
		pdf(paste0(plot.file,".pdf"), bg="white")
			plot(
				NULL, 
				main=paste0(g.names[i], " vs ", g.names[j]),
				xlab="Adaptive hard seeds", ylab="Exact matches",
				xlim=range(sn), ylim=range(c(all.evol))
			)
				
			# loop over matching methods and plot each as a series
			for(m in 1:length(methods))
				lines(x=sn, y=all.evol[,m], col=colors[m], lwd=2)
		
		# add legend
		legend(
			x="topleft",
			legend=methods,
			fill=colors
		)
		
		# close plot
		dev.off()
	}
}
