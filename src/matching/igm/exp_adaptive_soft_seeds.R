# Experiments with vertex matching, using the adaptive soft seed approach.
# It is similar to the adaptive hard seed method, except the soft method
# allows matching several candidates.
# 
# Author: Vincent Labatut
# 06/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/igm/exp_adaptive_soft_seeds.R")
###############################################################################
library("igraph")
library("iGraphMatch")
library("RColorBrewer")




###############################################################################
# processing parameters
MAX_ITER <- 200




###############################################################################
# output folder
out.folder <- file.path("out","matching")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)
mode.folder <- "common_raw_adaptive_soft"




###############################################################################
# load the static graphs

# notes:
# - comics:
#   - last chapter: #143 (file cum/inst_143.graphml)
#   - last scene: #1438 (file cum/inst_1437.graphml)
# - tv show:
#   - last S02 episode: #20 (file cumulative/instant_019.graphml)
#   - last S02 scene: #754 (file cumulative/instant_0753.graphml)
#   - very last episode: #73 (file cumulative/instant_072.graphml)
#   - very last scene: #4165 (file cumulative/instant_4164.graphml)
# - novels:
#   - last book 2 chapter: 2.ACoK_69_cumul.graphml / 2.ACoK_69_instant.graphml
#   - last chapter: 5.ADwD_72_cumul.graphml / 5.ADwD_72_instant.graphml

# read the chapter-based novel static graph
g.nv <- read.graph("in/novels/cumul/2.ACoK_69_cumul.graphml", format="graphml")
g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)			# keep only named characters
E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)			# normalize weights

# read the scene-based comics static graph
g.cx <- read.graph("in/comics/cumul/scene/cum_1437.graphml", format="graphml")
g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)			# keep only named characters
E(g.cx)$weight <- E(g.cx)$Occurrences/max(E(g.cx)$Occurrences)	# normalize weights

# read the episode-based tvshow static graph
g.tv <- read.graph("in/tvshow/cumul/scene/cumulative_0753.graphml", format="graphml")
g.tv <- delete_vertices(graph=g.tv, v=!V(g.tv)$named)			# keep only named characters
E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)			# normalize weights




###############################################################################
# identify most important characters (according to novels)
top.chars <- V(g.nv)$name[order(degree(g.nv),decreasing=TRUE)][1:20]




###############################################################################
# adaptive soft seeding
gs <- list(g.nv, g.cx, g.tv)
g.names <- c("novels","comics","tvshow")
methods <- c("convex", "indefinite", "PATH", "percolation", "IsoRank", "Umeyama")

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
		
		# loop over matching methods
		for(m in 1:length(methods))
		{	method <- methods[m]
			cat("......Applying method ",method,"\n",sep="")
			local.folder <- file.path(out.folder, mode.folder, comp.name, method)
			dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
			
			# loop over seed number
			tab.evol <- c()
			seeds <- NULL
			start_soft <- "bari"
			for(s in sn)
			{	cat("........Number of seeds: ",s,"\n",sep="")
				
				# update seeds
				if(s>0)
				{	bm <- best_matches(A=g1, B=g2, match=res, measure="row_cor")			# "row_cor", "row_diff", or "row_perm_stat"
					bm0 <- bm[!is.na(bm[,"A_best"]) & !is.na(bm[,"B_best"]),]
					seeds_bm <- head(bm0, min(s,nrow(bm0)))
					seeds <- seeds_bm[, 1:2]
					start_soft <- init_start(start="bari", nns=max(gorder(g1), gorder(g2)), soft_seeds=seeds)
					startm <- as.matrix(start_soft)
				}
				
				if(method=="indefinite")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						#similarity,			# vertex-vertex similarity matrix
						
						method="indefinite",	# matching method: indefinite relaxation of the objective function
						start=start_soft,		# initialization method for the matrix
						#lap_method=NULL,		# method used to solve LAP
						max_iter=MAX_ITER		# maximum number of replacing matches
					)
				}
				else if(method=="convex")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						#similarity,			# vertex-vertex similarity matrix
						
						method="convex",		# matching method: convex relaxation of the objective function
						start=start_soft,		# initialization method for the matrix
						#lap_method=NULL,		# method used to solve LAP
						max_iter=MAX_ITER,		# maximum number of replacing matches
						#tol = 1e-05			# tolerance of edge disagreements
					)
				}
				else if(method=="PATH")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						similarity=startm,		# vertex-vertex similarity matrix
						
						method="PATH",			# matching method: ?
						#lap_method=NULL,		# method used to solve LAP
						#epsilon=1,				# small value
						max_iter=MAX_ITER		# maximum number of replacing matches
					)
				}
				else if(method=="percolation")
				{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)
					res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seed,				# known vertex matches
						similarity=startm,		# vertex-vertex similarity matrix
						
						method="percolation",	# matching method: percolation
						#r="2",					# threshold of neighboring pair scores
						ExpandWhenStuck=TRUE	# expand the seed set when Percolation algorithm stops before matching all the vertices (better when few seeds)
					)
				}
				else if(method=="IsoRank")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						#seeds,					# known vertex matches
						similarity=startm,		# vertex-vertex similarity matrix
			
						method="IsoRank",		# matching method: IsoRank algorithm (spectral method)
						#lap_method=NULL,		# method used to solve LAP
						max_iter=MAX_ITER		# maximum number of replacing matches
					)
				}
				else if(method=="Umeyama")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						similarity=startm,		# vertex-vertex similarity matrix
						
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
				# same thing for the 20 top characters
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
					print(tab[1:20,])
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
		all.evol <- c()
		for(m in 1:length(methods))
		{	# read evolution table
			method <- methods[m]
			tab.evol <- read.csv(file=file.path(local.folder,method,"_exact_matches_evolution.csv"), header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
			all.evol <- c(all.evol, tab.evol[,"ExactMatches"])
		}
		
		# create plot
		plot.file <- file.path(local.folder,"exact_matches_evolution")
		pdf(paste0(plot.file,".pdf"), bg="white")
			plot(
				NULL, 
				main=paste0(g.names[i], " vs ", g.names[j]),
				xlab="Adaptive soft seeds", ylab="Exact matches",
				xlim=range(sn), ylim=range(all.evol)
			)
				
			# loop over matching methods
			for(m in 1:length(methods))
			{	# read evolution table
				method <- methods[m]
				tab.evol <- read.csv(file=file.path(local.folder,method,"_exact_matches_evolution.csv"), header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
				
				# plot values
				lines(x=tab.evol[,"AdaptiveSeeds"], y=tab.evol[,"ExactMatches"], col=colors[m], lwd=2)
			}
		
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





###############################################################################
# TODO
# - try the adaptive seed approach
# - use time (ie. the narrative dynamics)
