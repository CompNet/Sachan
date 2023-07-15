# Experiments with vertex matching, using the adaptive hard seed approach,
# but applied to (aligned) dynamic networks. We use the first network at the
# 
# 
# 
# Author: Vincent Labatut
# 07/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/igm/exp_adaptive_hard_seeds_temporal.R")
###############################################################################
library("igraph")
library("iGraphMatch")
library("scales")




###############################################################################
# processing parameters
MAX_ITER <- 200




###############################################################################
# output folder
out.folder <- file.path("out","matching")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)
mode.folder <- "common_raw_adaptive_hard_temporal"




###############################################################################
# load the dynamic graphs

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

# read all chapter-based novel cumulative graphs (until book 2)
gs.nv <- list()
files <- sort(list.files(path="in/novels/cumul", pattern=".+\\.graphml", full.names=TRUE))
i <- 1
while(files[i]!="in/novels/cumul/3.ASoS_00_cumul.graphml")
{	cat("..Loading file \"",files[i],"\"\n",sep="")
	g.nv <- read.graph(files[i], format="graphml")
	g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)			# keep only named characters
	E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)			# normalize weights
	gs.nv <- c(gs.nv, list(g.nv))
	i <- i + 1
}
	cat("Loaded a total of ",length(gs.nv)," novel networks\n",sep="")

# read all scene-based comics cumulative graphs
gs.cx <- list()
files <- sort(list.files(path="in/comics/cumul/chapter", pattern=".+\\.graphml", full.names=TRUE))
for(i in 1:length(files))
{	cat("..Loading file \"",files[i],"\"\n",sep="")
	g.cx <- read.graph(files[i], format="graphml")
	g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)			# keep only named characters
	E(g.cx)$weight <- E(g.cx)$Occurrences/max(E(g.cx)$Occurrences)	# normalize weights
	gs.cx <- c(gs.cx, list(g.cx))
	i <- i + 1
}
cat("Loaded a total of ",length(gs.cx)," comic networks\n",sep="")

# read all scene-based tvshow static graphs (until season 2)
#gs.tv <- list
#files <- sort(list.files(path="in/novels/cumul", pattern=".+\\.graphml", full.names=TRUE))
#i <- 1
#while(files[i]!="in/novels/cumul/2.ACoK_69_cumul.graphml")
#{	cat("..Loading file \"",files[i],"\"\n",sep="")
#	g.tv <- read.graph(file, format="graphml")
#	g.tv <- delete_vertices(graph=g.tv, v=!V(g.tv)$named)			# keep only named characters
#	E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)			# normalize weights
#	gs.tv <- c(gs.tv, list(g.tv))
#	i <- i + 1
#}
#cat("Loaded a total of ",length(gs.tv)," TV show networks\n",sep="")




###############################################################################
# load the chapter mapping file
tab <- read.csv(file="in/comics/chapters.csv", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
chap.map <- tab[,"Rank"]
# re-order comic networks ?
# TODO pb: need the cumulative nets built in the proper order




###############################################################################
# adaptive hard seeding
gs <- list(gs.nv, gs.cx)			# gs.tv
g.names <- c("novels","comics")		# "tvshow"
methods <- c("convex", "indefinite", "PATH", "percolation", "Umeyama")	# "IsoRank" requires a vertex similarity matrix

tab.exact.matches <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods))
colnames(tab.exact.matches) <- methods
rownames(tab.exact.matches) <- rep(NA,nrow(tab.exact.matches))
r <- 1

# loop over pairs of networks
for(i in 1:(length(gs)-1))
{	cat("..Processing first narrative ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second narrative ",g.names[j],"\n",sep="")
		
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		rownames(tab.exact.matches)[r] <- comp.name
		
		# loop over matching methods
		for(m in 1:length(methods))
		{	method <- methods[m]
			cat("......Applying method ",method,"\n",sep="")
			local.folder <- file.path(out.folder, mode.folder, comp.name, method)
			dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
			
			tab.evol <- c()
			seeds <- NULL
			sn <- 0
			
			# loop over time
			k <- 1
			while(k<=length(gs[[i]]) && k<=length(gs[[j]]))
			{	cat("........Processing iteration ",k,"\n",sep="")
				
				g1 <- gs[[i]][[k]]
				g2 <- gs[[j]][[k]]
				
				# focus on characters common to both networks
				names <- intersect(V(g1)$name,V(g2)$name)
				idx1 <- which(!(V(g1)$name %in% names))
				g1 <- delete_vertices(g1,idx1)
				idx2 <- which(!(V(g2)$name %in% names))
				g2 <- delete_vertices(g2,idx2)
				
				if(method=="indefinite")
				{	res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seeds,			# known vertex matches
						#similarity,			# vertex-vertex similarity matrix
						
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
						#similarity,			# vertex-vertex similarity matrix
						
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
						#similarity,			# vertex-vertex similarity matrix
						
						method="PATH",			# matching method: ?
						#lap_method=NULL,		# method used to solve LAP
						#epsilon=1,				# small value
						max_iter=MAX_ITER		# maximum number of replacing matches
					)
				}
				else if(method=="percolation")
				{	top.chars1 <- V(g1)$name[order(degree(g1),decreasing=TRUE)]
					top.chars2 <- V(g2)$name[order(degree(g2),decreasing=TRUE)]
					top.inter <- intersect(top.chars1,top.chars2)
					seed <- matrix(c(which(V(g1)$name==top.inter[1]),which(V(g2)$name==top.inter[1])), ncol=2)	# this method needs at least one seed
					# TODO what if the inter is empty ?
					if(!is.null(seeds) && nrow(seeds)>0)
						seed <- seeds
					res <- gm(
						A=g1, B=g2,				# graphs to compare 
						seeds=seed,				# known vertex matches
						#similarity,			# vertex-vertex similarity matrix
						
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
						#similarity,			# vertex-vertex similarity matrix
						
						method="Umeyama"		# matching method: Umeyama algorithm (spectral)
					)
				}
				
				# update seeds
				if(k<length(gs[[i]]) && k<length(gs[[j]]))
				{	if(method=="percolation" && k==1)
						bm <- best_matches(A=g1, B=g2, match=res, measure="row_diff")
					else
						bm <- best_matches(A=g1, B=g2, match=res, measure="row_cor")	# "row_cor", "row_diff", or "row_perm_stat"
					bm0 <- bm[!is.na(bm[,"A_best"]) & !is.na(bm[,"B_best"]),]
					seeds <- bm0[, 1:2]
					sn <- c(sn, nrow(seeds))
				}
				
				sink(file.path(local.folder,paste0("seeds",sprintf("%03d",k),"_summary.txt")))
				# ground truth (not useful due to the NAs resulting from different characters in the nets)
				gt <- match(V(g1)$name, V(g2)$name)
				# summary of the matching process
				tryCatch(expr={print(summary(object=res, A=g1, B=g2, true_label=gt))}, 
						error=function(e) {})
				# number of perfect matches
				idx.exact.matches <- which(V(g1)$name[res$corr_A]==V(g2)$name[res$corr_B])
				nbr.exact.matches <- length(idx.exact.matches)
				tab.exact.matches[r,method] <- nbr.exact.matches
				tab.evol <- c(tab.evol, nbr.exact.matches)
				cat("Number of perfect matches:",nbr.exact.matches,"\n")
				sink()
				tryCatch(expr={print(summary(object=res, A=g1, B=g2, true_label=gt))}, 
					error=function(e) {})
				cat("Number of perfect matches:",nbr.exact.matches,"\n")
				
				# list perfect matches
				if(nbr.exact.matches>0)
				{	exact.matches <- matrix(V(g1)$name[res$corr_A][idx.exact.matches],ncol=1)
					colnames(exact.matches) <- "Character"
					cat("List of exact matches:\n")
					print(exact.matches)
					write.csv(x=exact.matches, file=file.path(local.folder,paste0("seeds",sprintf("%03d",k),"_list_exact_matches.csv")), row.names=FALSE, fileEncoding="UTF-8")
				}
				
				# list all character matches
				tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])
				colnames(tab) <- c("Corr_A","Corr_B")
				#print(tab)
				write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%03d",k),"_list_full.csv")), row.names=FALSE, fileEncoding="UTF-8")
				# same thing for the 20 top characters
				tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])[V(g1)$name[res$corr_A] %in% top.chars | V(g2)$name[res$corr_B] %in% top.chars,]
				colnames(tab) <- c("Top_Corr_A","Top_Corr_B")
				print(tab)
				write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%03d",k),"_list_top.csv")), row.names=FALSE, fileEncoding="UTF-8")
				
				# best matches according to some internal criterion (no GT)
				for(meas in c("row_cor", "row_diff", "row_perm_stat"))
				{	cat(".........Computing measure ",meas,"\n",sep="")
					tryCatch(expr=
					{	bm <- best_matches(A=g1, B=g2, match=res, measure=meas)			# "row_cor", "row_diff", or "row_perm_stat"
						tab <- cbind(bm,V(g1)$name[bm$A_best], V(g2)$name[bm$B_best])
						colnames(tab)[(ncol(bm)+1):(ncol(bm)+2)] <- c("Character_A","Character_B")
						print(tab[1:20,])
					}, error=function(e) NA)
					write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%03d",k),"_best_matches_",meas,".csv")), row.names=FALSE, fileEncoding="UTF-8")
				}
				
				k <- k + 1
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
		sss <- min(length(gs[[i]]), length(gs[[j]]))
		
		# loop over matching methods
		all.evol <- matrix(NA,nrow=sss,ncol=length(methods))
		all.sn <- matrix(NA,nrow=sss,ncol=length(methods))
		for(m in 1:length(methods))
		{	# read evolution table
			method <- methods[m]
			tab.evol <- read.csv(file=file.path(local.folder,method,"_exact_matches_evolution.csv"), header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
			all.evol[,m] <- tab.evol[,"ExactMatches"]
			all.sn[,m] <- tab.evol[,"AdaptiveSeeds"]
		}
		
		# record evolution tables
		colnames(all.evol) <- methods
		write.csv(x=all.evol, file=file.path(local.folder,"exact_matches_evolution_values.csv"), row.names=TRUE, fileEncoding="UTF-8")
		colnames(all.sn) <- methods
		write.csv(x=all.sn, file=file.path(local.folder,"exact_matches_evolution._seeds.csv"), row.names=TRUE, fileEncoding="UTF-8")
		
		# create plot
		plot.file <- file.path(local.folder,"exact_matches_evolution")
		pdf(paste0(plot.file,".pdf"), bg="white")
		plot(
			NULL, 
			main=paste0(g.names[i], " vs ", g.names[j]),
			xlab="Adaptive hard seeds", ylab="Exact matches",
			xlim=range(c(all.sn)), ylim=range(c(all.evol))
		)
		
		# loop over matching methods and plot each as a series
		for(m in 1:length(methods))
			lines(x=all.sn[,m], y=all.evol[,m], col=colors[m], lwd=2)
		
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

# TODO
# - common_raw_noseed is missing
# - plots for temporal seeds are all messed up
# - update report with new results
# - try to integrate vertex attributes
###
# - descriptive instead: compute vertex sim distrib
