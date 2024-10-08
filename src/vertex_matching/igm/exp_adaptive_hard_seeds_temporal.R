# Experiments with vertex matching, using the adaptive hard seed approach,
# but applied to (aligned) dynamic networks. We use the first network at the
# 
# 
# 
# Author: Vincent Labatut
# 07/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/vertex_matching/igm/exp_adaptive_hard_seeds_temporal.R")
###############################################################################
library("igraph")
library("iGraphMatch")
library("scales")

source("src/common/stats.R")




###############################################################################
# processing parameters
MAX_ITER <- 200				# limit on the number of iterations during matching
TOP_CHAR_NBR <- 20			# number of important characters




###############################################################################
# output folder
out.folder <- file.path("out","vertex_matching","first_2","attr_none")
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)
mode.folder <- "common_raw_adaptive_hard_temporal"




###############################################################################
# no choice here, we need to use the cumulative networks
CUMULATIVE <- TRUE
# only take the first two narrative units (whole narrative not supported here)
NARRATIVE_PART <- 2
# load the dynamic graphs
NU_NV <- "chapter"	# narrative unit for the novels: only "chapter" is possible
NU_CX <- "chapter"	# narrative unit for the comics: only "chapter" is supported in this script
source("src/common/load_dynamic_nets.R")




###############################################################################
## read the list of characters ranked by importance
source("src/common/char_importance.R")
tab.file <- file.path("in",paste0("ranked_importance_S",NARRATIVE_PART,".csv"))
char.importance <- read.csv(file=tab.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE, fileEncoding="UTF-8")
ranked.chars <- char.importance[,"Name"]
imp.moy <- char.importance[,"Mean"]
names(imp.moy) <- char.importance[,"Name"]




###############################################################################
# adaptive hard seeding
gs <- list(gs.nv, gs.cx)			# gs.tv
g.names <- c("novels","comics")		# "tvshow"
methods <- c("convex", "indefinite", "PATH", "percolation", "Umeyama")	# "IsoRank" requires a vertex similarity matrix
m.names <- c("convex"="Convex", "indefinite"="Indefinite", "PATH"="Concave", "percolation"="Percolation", "Umeyama"="Umeyama", "IsoRank"="IsoRank")

tab.exact.matches <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods)+1)
colnames(tab.exact.matches) <- c(methods,"CharNbr")
rownames(tab.exact.matches) <- rep(NA,nrow(tab.exact.matches))
tab.evol.matches <- matrix(NA,nrow=length(g.names)*(length(g.names)-1),ncol=length(methods))
colnames(tab.evol.matches) <- methods
rownames(tab.evol.matches) <- rep(NA,nrow(tab.evol.matches))
r <- 1

# loop over pairs of networks
for(i in 1:(length(gs)-1))
{	cat("..Processing first narrative ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second narrative ",g.names[j],"\n",sep="")
		
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		comp.name.rev <- paste0(g.names[j], "_vs_", g.names[i])
		rownames(tab.exact.matches)[r] <- comp.name
		rownames(tab.evol.matches)[2*r-1] <- comp.name
		rownames(tab.evol.matches)[2*r] <- comp.name.rev
		
		# loop over matching methods
		all.chr.names <- c()
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
				chr.names <- intersect(V(g1)$name,V(g2)$name)
				all.chr.names <- union(all.chr.names,chr.names)
				idx1 <- which(!(V(g1)$name %in% chr.names))
				g1 <- delete_vertices(g1,idx1)
				idx2 <- which(!(V(g2)$name %in% chr.names))
				g2 <- delete_vertices(g2,idx2)
				
				# indentify top characters
				top.chars <- setdiff(ranked.chars, setdiff(ranked.chars, union(V(g1)$name,V(g2)$name)))
				
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
				{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)	# this method needs at least one seed
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
				
				# update match matrices
				if(k==1)
				{	match.evol.mtrx1 <- matrix(V(g2)$name[res$corr_B],nrow=1)
					colnames(match.evol.mtrx1) <- V(g1)$name[res$corr_A]
					match.evol.mtrx2 <- matrix(V(g1)$name[res$corr_A], nrow=1)
					colnames(match.evol.mtrx2) <- V(g2)$name[res$corr_B]
				}
				else
				{	new.names <- setdiff(V(g1)$name[res$corr_A], colnames(match.evol.mtrx1))
					old.names <- colnames(match.evol.mtrx1)
					match.evol.mtrx1 <- cbind(match.evol.mtrx1, matrix(NA,nrow=nrow(match.evol.mtrx1),ncol=length(new.names)))
					colnames(match.evol.mtrx1) <- c(old.names, new.names)
					match.evol.mtrx1 <- rbind(match.evol.mtrx1, rep(NA,ncol(match.evol.mtrx1)))
					match.evol.mtrx1[nrow(match.evol.mtrx1),V(g1)$name[res$corr_A]] <- V(g2)$name[res$corr_B]
					#
					new.names <- setdiff(V(g2)$name[res$corr_B], colnames(match.evol.mtrx2))
					old.names <- colnames(match.evol.mtrx2)
					match.evol.mtrx2 <- cbind(match.evol.mtrx2, matrix(NA,nrow=nrow(match.evol.mtrx2),ncol=length(new.names)))
					colnames(match.evol.mtrx2) <- c(old.names, new.names)
					match.evol.mtrx2 <- rbind(match.evol.mtrx2, rep(NA,ncol(match.evol.mtrx2)))
					match.evol.mtrx2[nrow(match.evol.mtrx2),V(g2)$name[res$corr_B]] <- V(g1)$name[res$corr_A]
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
				# same thing for the most important characters
				tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])[V(g1)$name[res$corr_A] %in% top.chars | V(g2)$name[res$corr_B] %in% top.chars,,drop=FALSE]
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
						print(tab[1:TOP_CHAR_NBR,])
					}, error=function(e) NA)
					write.csv(x=tab, file=file.path(local.folder,paste0("seeds",sprintf("%03d",k),"_best_matches_",meas,".csv")), row.names=FALSE, fileEncoding="UTF-8")
				}
				
				k <- k + 1
			}
			
			# record evolution table
			tab.evol <- data.frame(sn, tab.evol)
			colnames(tab.evol) <- c("AdaptiveSeeds","ExactMatches")
			write.csv(x=tab.evol, file=file.path(local.folder,"_exact_matches_evolution.csv"), row.names=FALSE, fileEncoding="UTF-8")
			
			# take the modal match for each character
			dyn.matches1 <- apply(match.evol.mtrx1, 2, function(col) mode(col, na.rm=TRUE))
			perf.evol1 <- length(which(dyn.matches1==names(dyn.matches1)))/length(dyn.matches1)
			dyn.matches2 <- apply(match.evol.mtrx2, 2, function(col) mode(col, na.rm=TRUE))
			perf.evol2 <- length(which(dyn.matches2==names(dyn.matches2)))/length(dyn.matches2)
			tab.evol.matches[2*r-1,method] <- perf.evol1
			tab.evol.matches[2*r,method] <- perf.evol2
		}
		
		# update perf table
		tab.exact.matches[r,"CharNbr"] <- length(all.chr.names)
		
		r <- r + 1
	}
}

# record overall tables
print(tab.exact.matches)
write.csv(x=tab.exact.matches, file=file.path(out.folder,mode.folder,"exact_matches_comparison.csv"), row.names=TRUE, fileEncoding="UTF-8")
print(tab.evol.matches)
write.csv(x=tab.evol.matches, file=file.path(out.folder,mode.folder,"exact_matches_modal.csv"), row.names=TRUE, fileEncoding="UTF-8")




###############################################################################
# plot the evolution of the performance over iterations
#colors <- 1:length(methods)
colors <- brewer_pal(type="qual", palette=2)(length(methods))

for(i in 1:(length(gs)-1))
{	for(j in (i+1):length(gs))
	{	comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		comp.title <- bquote(bolditalic(.(narr.names[g.names[i]]))~bold(" vs. ")~bolditalic(.(narr.names[g.names[j]])))
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
		
		### create performance plot
		plot.file <- file.path(local.folder,"exact_matches_evolution")
		pdf(paste0(plot.file,".pdf"))	# bg="white"
			plot(
				NULL, 
				main=comp.title,
				xlab="Time", ylab="Exact matches",
				xlim=c(1,nrow(all.evol)), ylim=range(c(all.evol))
			)
			# loop over matching methods and plot each as a series
			for(m in 1:length(methods))
				lines(x=1:nrow(all.evol), y=all.evol[,m], col=colors[m], lwd=2)
			# add legend
			legend(
				x="topleft",
				legend=m.names[methods],
				fill=colors,
				bg="#FFFFFFCC"
			)
		# close plot
		dev.off()
		
		### create seed number plot
		plot.file <- file.path(local.folder,"seed_number_evolution")
		pdf(paste0(plot.file,".pdf"))	# bg="white"
			plot(
				NULL, 
				main=comp.title,
				xlab="Time", ylab="Seed number",
				xlim=c(1,nrow(all.sn)), ylim=range(c(all.sn))
			)
			# loop over matching methods and plot each as a series
			for(m in 1:length(methods))
				lines(x=1:nrow(all.sn), y=all.sn[,m], col=colors[m], lwd=2)
			# add legend
			legend(
				x="topleft",
				legend=m.names[methods],
				fill=colors,
				bg="#FFFFFFCC"
			)
		# close plot
		dev.off()
	}
}
