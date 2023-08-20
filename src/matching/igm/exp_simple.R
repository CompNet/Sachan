# Experiments with vertex matching, using the standard approach.
# We experiment using seeds, focusing on common characters (between
# both compared graphs), centering the graphs (some preprocessing
# supposed to improve the matching). 
# 
# Author: Vincent Labatut
# 04/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/igm/exp_simple.R")
###############################################################################
library("igraph")
library("iGraphMatch")




###############################################################################
# processing parameters
MAX_ITER <- 200				# limit on the number of iterations during matching
COMMON_CHARS_ONLY <- FALSE	# all named characters, or only those common to both compared graphs
CENTER_GRAPHS <- FALSE		# whether to perform the centering preprocessing step
USE_SEEDS <- FALSE			# whether to use seeds to bootstrap the matching process
USE_SEEDS_NBR <- 15			# number of seeds used (if any)
ATTR <- "both"				# attribute used during matching: none sex affiliation both
WHOLE_NARRATIVE <- FALSE	# only take the first two books, all comics, first two seasons (whole narrative not supported here)




###############################################################################
# output folder
out.folder <- file.path("out","matching",paste0("attr_",ATTR))
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)

{	if(COMMON_CHARS_ONLY)
		mode.folder <- "common"
	else
		mode.folder <- "named"
	
	if(CENTER_GRAPHS)
		mode.folder <- paste0(mode.folder, "_centered")
	else
		mode.folder <- paste0(mode.folder, "_raw")
	
	if(USE_SEEDS)
		mode.folder <- paste0(mode.folder, "_",USE_SEEDS_NBR,"seeds")
	else
		mode.folder <- paste0(mode.folder, "_noseeds")
}




###############################################################################
# load the static graphs
source("src/common/load_static_nets.R")




###############################################################################
# retrieve the characters' affiliations
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
# clean up a bit
aff.map <- char.tab[,"AllegianceBoth"]
names(aff.map) <- char.tab[,"Name"]
aff.map[aff.map==""] <- "Unknown"
aff.map <- sapply(strsplit(x=aff.map, split=",", fixed=TRUE), function(v) v[1])

# add to novel network
aff <- aff.map[V(g.nv)$name]
aff[is.na(aff)] <- "Unknown"
V(g.nv)$affiliation <- aff
# add to comics network
aff <- aff.map[V(g.cx)$name]
aff[is.na(aff)] <- "Unknown"
V(g.cx)$affiliation <- aff
# add to TV show network
aff <- aff.map[V(g.tv)$name]
aff[is.na(aff)] <- "Unknown"
V(g.tv)$affiliation <- aff




###############################################################################
# identify most important characters (according to novels)
top.chars <- V(g.nv)$name[order(degree(g.nv),decreasing=TRUE)][1:20]
char.seeds <- top.chars[1:USE_SEEDS_NBR]




###############################################################################
# start matching
gs <- list(g.nv, g.cx, g.tv)
g.names <- c("novels","comics","tvshow")
methods <- c("convex", "indefinite", "PATH", "percolation", "Umeyama")	# "IsoRank" requires a vertex similarity matrix

tab.exact.matches <- matrix(NA,nrow=length(g.names)*(length(g.names)-1)/2,ncol=length(methods))
colnames(tab.exact.matches) <- methods
rownames(tab.exact.matches) <- rep(NA,nrow(tab.exact.matches))
r <- 1

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
		if(COMMON_CHARS_ONLY)
		{	names <- intersect(V(g1)$name,V(g2)$name)
			idx1 <- which(!(V(g1)$name %in% names))
			g1 <- delete_vertices(g1,idx1)
			idx2 <- which(!(V(g2)$name %in% names))
			g2 <- delete_vertices(g2,idx2)
		}
		
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
		
		# possibly center the graphs
		dg1 <- g1
		dg2 <- g2
		if(CENTER_GRAPHS)
		{	dg1 <- center_graph(g1, scheme="center", use_splr=TRUE)
			dg2 <- center_graph(g2, scheme="center", use_splr=TRUE)
		}
		# possibly handle seeds
		seeds <- NULL
		if(USE_SEEDS)
			seeds <- cbind(match(char.seeds,V(g1)$name), match(char.seeds,V(g2)$name))
		
		# loop over matching methods
		for(m in 1:length(methods))
		{	method <- methods[m]
			cat("......Applying method ",method,"\n",sep="")
			local.folder <- file.path(out.folder, mode.folder, comp.name, method)
			dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
			
			if(method=="indefinite")
			{	res <- gm(
					A=dg1, B=dg2,			# graphs to compare 
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
					A=dg1, B=dg2,			# graphs to compare 
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
					A=dg1, B=dg2,			# graphs to compare 
					seeds=seeds,			# known vertex matches
					similarity=sim.mat,		# vertex-vertex similarity matrix
					
					method="PATH",			# matching method: ?
					#lap_method=NULL,		# method used to solve LAP
					#epsilon=1,				# small value
					max_iter=MAX_ITER		# maximum number of replacing matches
				)
			}
			else if(method=="percolation")
			{	seed <- matrix(c(which(V(g1)$name==top.chars[1]),which(V(g2)$name==top.chars[1])), ncol=2)
				if(USE_SEEDS)
					seed <- seeds
				res <- gm(
					A=dg1, B=dg2,			# graphs to compare 
					seeds=seed,				# known vertex matches
					similarity=sim.mat,		# vertex-vertex similarity matrix
					
					method="percolation",	# matching method: percolation
					#r="2",					# threshold of neighboring pair scores
					ExpandWhenStuck=TRUE	# expand the seed set when Percolation algorithm stops before matching all the vertices (better when few seeds)
				)
			}
#			else if(method=="IsoRank")
#			{	res <- gm(
#					A=dg1, B=dg2,			# graphs to compare 
#					#seeds,					# known vertex matches
#					similarity=,			# vertex-vertex similarity matrix (required for method "IsoRank")
#					
#					method="IsoRank",		# matching method: IsoRank algorithm (spectral method)
#					#lap_method=NULL,		# method used to solve LAP
#					max_iter=MAX_ITER		# maximum number of replacing matches
#				)
#			}
			else if(method=="Umeyama")
			{	res <- gm(
					A=dg1, B=dg2,			# graphs to compare 
					seeds=seeds,			# known vertex matches
					similarity=sim.mat,		# vertex-vertex similarity matrix
					
					method="Umeyama"		# matching method: Umeyama algorithm (spectral)
				)
			}
			
			sink(file.path(local.folder,"summary.txt"))
			# ground truth (not useful due to the NAs resulting from different characters in the nets)
			gt <- match(V(g1)$name, V(g2)$name)
			# summary of the matching process
			print(summary(res, dg1, dg2, true_label=gt))
			# number of perfect matches
			idx.exact.matches <- which(V(g1)$name[res$corr_A]==V(g2)$name[res$corr_B])
			nbr.exact.matches <- length(idx.exact.matches)
			tab.exact.matches[r,method] <- nbr.exact.matches
			cat("Number of perfect matches:",nbr.exact.matches,"\n")
			sink()
			print(summary(res, dg1, dg2, true_label=gt))
			cat("Number of perfect matches:",nbr.exact.matches,"\n")
			
			# list perfect matches
			if(nbr.exact.matches>0)
			{	exact.matches <- matrix(V(g1)$name[res$corr_A][idx.exact.matches],ncol=1)
				colnames(exact.matches) <- "Character"
				cat("List of exact matches:\n")
				print(exact.matches)
				write.csv(x=exact.matches, file=file.path(local.folder,"list_exact_matches.csv"), row.names=FALSE, fileEncoding="UTF-8")
			}
			
			# list all character matches
			tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])
			colnames(tab) <- c("Corr_A","Corr_B")
			#print(tab)
			write.csv(x=tab, file=file.path(local.folder,"list_full.csv"), row.names=FALSE, fileEncoding="UTF-8")
			# same thing for the 20 top characters
			tab <- cbind(V(g1)$name[res$corr_A], V(g2)$name[res$corr_B])[V(g1)$name[res$corr_A] %in% top.chars | V(g2)$name[res$corr_B] %in% top.chars,]
			colnames(tab) <- c("Top_Corr_A","Top_Corr_B")
			print(tab)
			write.csv(x=tab, file=file.path(local.folder,"list_top.csv"), row.names=FALSE, fileEncoding="UTF-8")
			
			# best matches according to some internal criterion (no GT)
			for(meas in c("row_cor", "row_diff", "row_perm_stat"))
			{	cat(".........Computing measure ",meas,"\n",sep="")
				bm <- best_matches(A=g1, B=g2, match=res, measure=meas)			# "row_cor", "row_diff", or "row_perm_stat"
				tab <- cbind(bm,V(g1)$name[bm$A_best], V(g2)$name[bm$B_best])
				colnames(tab)[(ncol(bm)+1):(ncol(bm)+2)] <- c("Character_A","Character_B")
				print(tab[1:20,])
				write.csv(x=tab, file=file.path(local.folder,paste0("best_matches_",meas,".csv")), row.names=FALSE, fileEncoding="UTF-8")
			}
		}
		
		r <- r + 1
	}
}

# record overall table
print(tab.exact.matches)
write.csv(x=tab.exact.matches, file=file.path(out.folder,mode.folder,"exact_matches_comparison.csv"), row.names=TRUE, fileEncoding="UTF-8")
