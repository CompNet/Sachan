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
library("scales")




###############################################################################
# processing parameters
MAX_ITER <- 200				# limit on the number of iterations during matching
NARRATIVE_PART <- 2			# take the first two (2) or five (5) narrative units
COMMON_CHARS_ONLY <- TRUE	# all named characters (FALSE), or only those common to both compared graphs (TRUE)
TOP_CHAR_NBR <- 20			# number of important characters




###############################################################################
# output folder
out.folder <- file.path("out", "matching")
{	if(NARRATIVE_PART==0)
		out.folder <- file.path(out.folder, "whole_narr")
	else if(NARRATIVE_PART==2)
		out.folder <- file.path(out.folder, "first_2")
	else if(NARRATIVE_PART==5)
		out.folder <- file.path(out.folder, "first_5")
}
out.folder <- file.path(out.folder, "attr_none")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)

{	if(COMMON_CHARS_ONLY)
		mode.folder <- "common_raw_adaptive_soft"
	else
		mode.folder <- "named_raw_adaptive_soft"
}




###############################################################################
# load the static graphs
source("src/common/load_static_nets.R")




###############################################################################
# identify most important characters (according to novels)
top.chars <- ranked.chars[1:TOP_CHAR_NBR]




###############################################################################
# adaptive soft seeding
methods <- c("convex", "indefinite", "PATH", "percolation", "Umeyama", "IsoRank")
m.names <- c("convex"="Convex", "indefinite"="Indefinite", "PATH"="Concave", "percolation"="Percolation", "Umeyama"="Umeyama", "IsoRank"="IsoRank")

tab.exact.matches.all <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods)+1)
colnames(tab.exact.matches.all) <- c(methods,"CharNbr")
rownames(tab.exact.matches.all) <- rep(NA,nrow(tab.exact.matches.all))
tab.exact.matches.top <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods)+1)
colnames(tab.exact.matches.top) <- c(methods,"CharNbr")
rownames(tab.exact.matches.top) <- rep(NA,nrow(tab.exact.matches.top))
r <- 1
sn.all <- c(0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 130)	# numbers of seeds (all chars)
sn.top <- c(0:15)															# numbers of seeds (only top chars)

# loop over pairs of networks
for(i in 1:(length(gs)-1))
{	cat("..Processing first network ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second network ",g.names[j],"\n",sep="")
		g1 <- gs[[i]]
		g2 <- gs[[j]]
		
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		rownames(tab.exact.matches.all)[r] <- comp.name
		rownames(tab.exact.matches.top)[r] <- comp.name
		
		# focus on characters common to both networks
		if(COMMON_CHARS_ONLY)
		{	names <- intersect(V(g1)$name,V(g2)$name)
			idx1 <- which(!(V(g1)$name %in% names))
			g1 <- delete_vertices(g1,idx1)
			idx2 <- which(!(V(g2)$name %in% names))
			g2 <- delete_vertices(g2,idx2)
		}
		
		# first take all characters, then only top ones
		for(tc in c(FALSE,TRUE))
		{	# possibly reduce the graphs to the top characters
			if(tc)
			{	chars <- setdiff(ranked.chars, setdiff(ranked.chars, union(V(g1)$name,V(g2)$name)))
				idx1 <- match(setdiff(V(g1)$name, chars[1:TOP_CHAR_NBR]), V(g1)$name)
				g1 <- delete_vertices(graph=g1, v=idx1)
				idx2 <- match(setdiff(V(g2)$name, chars[1:TOP_CHAR_NBR]), V(g2)$name)
				g2 <- delete_vertices(graph=g2, v=idx2)
				# 
				tab.exact.matches <- tab.exact.matches.top
				suffx <- "_top"
				sn <- sn.top
			}
			else
			{	tab.exact.matches <- tab.exact.matches.all
				suffx <- "_all"
				sn <- sn.all
			}

			# update perf table
			char.nbr <- length(union(V(g1)$name,V(g2)$name))
			tab.exact.matches[r,"CharNbr"] <- char.nbr
			
			# loop over matching methods
			for(m in 1:length(methods))
			{	method <- methods[m]
				cat("......Applying method ",method,"\n",sep="")
				local.folder <- file.path(out.folder, mode.folder, comp.name, method)
				dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
				
				if(method=="indefinite" && tc)		# does not work, for some reason
					tab.evol <- rep(NA,length(sn))
				else
				{	# loop over seed number
					tab.evol <- c()
					seeds <- NULL
					startm <- NULL
					start_soft <- "bari"
					for(s in sn)
					{	cat("........Number of seeds: ",s,"\n",sep="")
						
						# update seeds
						if(s>0)
						{	bm <- best_matches(A=g1, B=g2, match=res, measure="row_cor")			# "row_cor", "row_diff", or "row_perm_stat"
							bm0 <- bm[!is.na(bm[,"A_best"]) & !is.na(bm[,"B_best"]),]
							seeds_bm <- head(bm0, min(s,nrow(bm0)))
							seeds <- seeds_bm[, 1:2,drop=FALSE]
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
						{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)	# this method needs at least one seed
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
						{	if(is.null(startm))
							{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)	# this method needs at least one seed
								start_soft <- init_start(start="bari", nns=max(gorder(g1), gorder(g2)), soft_seeds=seed)
								startm <- as.matrix(start_soft)
							}
							res <- gm(
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
						
						sink(file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_summary",suffx,".txt")))
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
							write.csv(x=exact.matches, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_list_exact_matches",suffx,".csv")), row.names=FALSE, fileEncoding="UTF-8")
						}
						
						# list all character matches
						tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])
						colnames(tab) <- c("Corr_A","Corr_B")
						#print(tab)
						write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_list_full",suffx,".csv")), row.names=FALSE, fileEncoding="UTF-8")
						# same thing for the most important characters
						if(!tc)
						{	tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])[V(g1)$name[res$corr_A] %in% top.chars | V(g2)$name[res$corr_B] %in% top.chars,]
							colnames(tab) <- c("Top_Corr_A","Top_Corr_B")
							print(tab)
							write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_list_top.csv")), row.names=FALSE, fileEncoding="UTF-8")
						}
						
						# best matches according to some internal criterion (no GT)
						if(!tc)
						{	for(meas in c("row_cor", "row_diff", "row_perm_stat"))
							{	cat(".........Computing measure ",meas,"\n",sep="")
								bm <- best_matches(A=g1, B=g2, match=res, measure=meas)			# "row_cor", "row_diff", or "row_perm_stat"
								tab <- cbind(bm,V(g1)$name[bm$A_best], V(g2)$name[bm$B_best])
								colnames(tab)[(ncol(bm)+1):(ncol(bm)+2)] <- c("Character_A","Character_B")
								print(tab[1:TOP_CHAR_NBR,])
								write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%02d",s),"_best_matches_",meas,".csv")), row.names=FALSE, fileEncoding="UTF-8")
							}
						}
					}
				}
				
				# record evolution table
				tab.evol <- data.frame(sn, tab.evol)
				colnames(tab.evol) <- c("AdaptiveSeeds","ExactMatches")
				write.csv(x=tab.evol, file=file.path(local.folder,paste0("_exact_matches_evolution",suffx,".csv")), row.names=FALSE, fileEncoding="UTF-8")
			}
					
			# update perf table
			if(tc)
				tab.exact.matches.top <- tab.exact.matches
			else
				tab.exact.matches.all <- tab.exact.matches
		}
		
		r <- r + 1
	}
}

# record overall table
cat("Overall results for all characters:\n")
print(tab.exact.matches.all)
write.csv(x=tab.exact.matches.all, file=file.path(out.folder,mode.folder,"exact_matches_comparison_all_counts.csv"), row.names=TRUE, fileEncoding="UTF-8")
tab.exact.matches.all.prop <- t(apply(tab.exact.matches.all, 1, function(row) row[1:(length(row)-1)]/row[length(row)]))
print(tab.exact.matches.all.prop)
write.csv(x=tab.exact.matches.all.prop, file=file.path(out.folder,mode.folder,"exact_matches_comparison_all_prop.csv"), row.names=TRUE, fileEncoding="UTF-8")
#
cat("Overall results for top characters:\n")
print(tab.exact.matches.top)
write.csv(x=tab.exact.matches.top, file=file.path(out.folder,mode.folder,"exact_matches_comparison_top_counts.csv"), row.names=TRUE, fileEncoding="UTF-8")
tab.exact.matches.top.prop <- t(apply(tab.exact.matches.top, 1, function(row) row[1:(length(row)-1)]/row[length(row)]))
print(tab.exact.matches.top.prop)
write.csv(x=tab.exact.matches.top.prop, file=file.path(out.folder,mode.folder,"exact_matches_comparison_top_prop.csv"), row.names=TRUE, fileEncoding="UTF-8")




###############################################################################
# plot the evolution of the performance over iterations
#colors <- 1:length(methods)
colors <- brewer_pal(type="qual", palette=2)(length(methods))

# first take all characters, then only top ones
for(tc in c(FALSE,TRUE))
{	# possibly reduce the graphs to the top characters
	if(tc)
	{	suffx <- "_top"
		sn <- sn.top
	}
	else
	{	suffx <- "_all"
		sn <- sn.all
	}
	for(i in 1:(length(gs)-1))
	{	for(j in (i+1):length(gs))
		{	comp.name <- paste0(g.names[i], "_vs_", g.names[j])
			comp.title <- bquote(bolditalic(.(narr.names[g.names[i]]))~bold(" vs. ")~bolditalic(.(narr.names[g.names[j]])))
			local.folder <- file.path(out.folder, mode.folder, comp.name)
			
			# loop over matching methods
			all.evol <- matrix(NA,nrow=length(sn),ncol=length(methods))
			for(m in 1:length(methods))
			{	# read evolution table
				method <- methods[m]
				tab.evol <- read.csv(file=file.path(local.folder,method,paste0("_exact_matches_evolution",suffx,".csv")), header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
				all.evol[,m] <- tab.evol[,"ExactMatches"]
			}
			
			# record evolution table
			tab <- data.frame(sn, all.evol)
			colnames(tab) <- c("AdaptiveSeeds", methods)
			write.csv(x=all.evol, file=file.path(local.folder,paste0("exact_matches_evolution",suffx,".csv")), row.names=TRUE, fileEncoding="UTF-8")
			
			# create plot
			plot.file <- file.path(local.folder,paste0("exact_matches_evolution",suffx))
			pdf(paste0(plot.file,".pdf"))	# bg="white"
				plot(
					NULL, 
					main=comp.title,
					xlab="Adaptive soft seeds", ylab="Exact matches",
					xlim=range(sn), ylim=range(c(all.evol),na.rm=TRUE)
				)
					
				# loop over matching methods and plot each as a series
				for(m in 1:length(methods))
					lines(x=sn, y=all.evol[,m], col=colors[m], lwd=2)
			
			# add legend
			legend(
				x="topleft",
				legend=m.names[methods],
				fill=colors
			)
			
			# close plot
			dev.off()
		}
	}
}
