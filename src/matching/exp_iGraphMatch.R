# Experiments with vertex matching.
# 
# Author: Vincent Labatut
# 04/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/exp_iGraphMatch.R.R")
###############################################################################
library("igraph")
library("iGraphMatch")




###############################################################################
# output folder
out.folder <- file.path("out","matching")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)




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
gs <- list(g.nv, g.cx, g.tv)
g.names <- c("novels","comics","tvshow")
methods <- c("indefinite", "convex", "PATH", "Umeyama")	# "IsoRank" requires a vertex similarity matrix, "percolation" results in a matrix error

tab.exact.matches <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods))
r <- 1

for(i in 1:(length(gs)-1))
{	cat("..Processing first network ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second network ",g.names[j],"\n",sep="")
		g1 <- gs[[i]]
		g2 <- gs[[j]]
		
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		rownames(tab.exact.matches)[r] <- comp.name
		
		for(m in 1:length(methods))
		{	method <- methods[m]
			cat("......Applying method ",method,"\n",sep="")
			local.folder <- file.path(out.folder, comp.name, method)
			dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
			
			if(method=="indefinite")
			{	res <- gm(
					A=g1, B=g2,				# graphs to compare 
					#seeds,					# known vertex matches
					#similarity,			# vertex-vertex similarity matrix
					
					method="indefinite",	# matching method: indefinite relaxation of the objective function
					start="bari", 			# initialization method for the matrix
					#lap_method=NULL,		# method used to solve LAP
					max_iter=200			# maximum number of replacing matches
				)
			}
			else if(method=="convex")
			{	res <- gm(
					A=g1, B=g2,				# graphs to compare 
					#seeds,					# known vertex matches
					#similarity,			# vertex-vertex similarity matrix
					
					method="convex",		# matching method: convex relaxation of the objective function
					start="bari", 			# initialization method for the matrix
					#lap_method=NULL,		# method used to solve LAP
					max_iter=200,			# maximum number of replacing matches
					#tol = 1e-05			# tolerance of edge disagreements
				)
			}
			else if(method=="PATH")
			{	res <- gm(
					A=g1, B=g2,				# graphs to compare 
					#seeds,					# known vertex matches
					#similarity,			# vertex-vertex similarity matrix
					
					method="PATH",			# matching method: ?
					#lap_method=NULL,		# method used to solve LAP
					#epsilon=1,				# small value
					max_iter=200			# maximum number of replacing matches
				)
			}
#			else if(method=="percolation")
#			{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)
#				res <- gm(
#					A=g1, B=g2,				# graphs to compare 
#					seeds=seed,				# known vertex matches
#					#similarity,			# vertex-vertex similarity matrix
#					
#					method="percolation",	# matching method: percolation
#					#r="2",					# threshold of neighboring pair scores
#					#ExpandWhenStuck=FALSE	# expand the seed set when Percolation algorithm stops before matching all the vertices
#				)
#			}
#			else if(method=="IsoRank")
#			{	res <- gm(
#					A=g1, B=g2,				# graphs to compare 
#					#seeds,					# known vertex matches
#					similarity=,			# vertex-vertex similarity matrix (required for method "IsoRank")
#					
#					method="IsoRank",		# matching method: IsoRank algorithm (spectral method)
#					#lap_method=NULL,		# method used to solve LAP
#					max_iter=200			# maximum number of replacing matches
#				)
#			}
			else if(method=="Umeyama")
			{	res <- gm(
					A=g1, B=g2,				# graphs to compare 
					#seeds,					# known vertex matches
					#similarity,			# vertex-vertex similarity matrix
					
					method="Umeyama"		# matching method: Umeyama algorithm (spectral)
				)
			}
			
			sink(file.path(local.folder,"summary.txt"))
			# ground truth (not useful due to the NAs resulting from different characters in the nets)
			gt <- match(V(g1)$name, V(g2)$name)
			# summary of the matching process
			print(summary(res, g1, g2, true_label=gt))
			# number of perfect matches
			idx.exact.matches <- which(V(g1)$name[res$corr_A]==V(g2)$name[res$corr_B])
			nbr.exact.matches <- length(idx.exact.matches)
			rownames(tab.exact.matches)[r,method] <- nbr.exact.matches
			cat("Number of perfect matches:",nbr.exact.matches,"\n")
			sink()
			
			# list perfect matches
			if(nbr.exact.matches>0)
			{	exact.matches <- matrix(V(g1)$name[res$corr_A][idx.exact.matches],ncol=1)
				cat("List of exact matches:\n")
				print(exact.matches)
				write.csv(x=exact.matches, file=file.path(local.folder,"list_exact_matches.csv"), row.names=FALSE, fileEncoding="UTF-8")
			}
			
			# list all character matches
			tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])
			#print(tab)
			write.csv(x=tab, file=file.path(local.folder,"list_full.csv"), row.names=FALSE, fileEncoding="UTF-8")
			# same thing for the 20 top characters
			tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])[V(g1)$name[res$corr_A] %in% top.chars | V(g2)$name[res$corr_B] %in% top.chars,]
			print(tab)
			write.csv(x=tab, file=file.path(local.folder,"list_top.csv"), row.names=FALSE, fileEncoding="UTF-8")
			
			# best matches according to some internal criterion (no GT)
			for(meas in c("row_cor", "row_diff", "row_perm_stat"))
			{	cat(".........Computing measure ",meas,"\n",sep="")
				bm <- best_matches(A=g1, B=g2, match=res, measure=meas)			# "row_cor", "row_diff", or "row_perm_stat"
				tab <- cbind(bm,V(g1)$name[bm$A_best], V(g2)$name[bm$B_best])
				print(tab[1:20,])
				write.csv(x=tab, file=file.path(local.folder,paste0("best_matches_",meas,".csv")), row.names=FALSE, fileEncoding="UTF-8")
			}
			
			r <- r + 1
		}
	}
}
