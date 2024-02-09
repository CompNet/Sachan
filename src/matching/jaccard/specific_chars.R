# Reports matching performances obtained for certain characters of interest.
# 
# Author: Vincent Labatut
# 02/2024
# 
# setwd("D:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/jaccard/specific_chars.R")
###############################################################################
library("igraph")




###############################################################################
# processing parameters
TOP_CHAR_NBR <- 20			# number of important characters (fixed)




###############################################################################
merged.chars <- matrix(c(
#	"Lyn Corbray","Lucas Corbray",			# ERROR: Lucas is actually missing from our novels annotations
	"Harrion Karstark","Arnolf Karstark", 
	"Benerro","Moqorro",
	"Meribald","Elder Brother"
),ncol=2,byrow=TRUE)
rownames(merged.chars) <- c(
#	"Vance Corbray",
	"Harald Karstark",
	"Kinvara",
	"Ray"
)
#
subs.chars <- c(
	"Jeyne Westerling"="Talisa Stark",
	"Vargo Hoat"="Locke",
	"Cleos Frey"="Alton Lannister",
	"Asha Greyjoy"="Yara Greyjoy",
	"Robert Arryn"="Robin Arryn",
	"Jhogo"="Kovarro",
	"Grazdan mo Eraz"="Razdal mo Eraz",
	"Grazdan mo Ullhor"="Greizhen mo Ullhor",
	"Stalwart Shield"="White Rat"
)




###############################################################################
# first approach: using the already computed similarity scores for named characters
for(NARRATIVE_PART in c(2,5))
{	cat(">>>>>>>>>> Processing narrative part = ",NARRATIVE_PART," <<<<<<<<<<\n",sep="")
	
	# load the csv file
	tab.file <- file.path("out", "matching", paste0("first_",NARRATIVE_PART), "jaccard", "named", "novels_vs_tvshow", "sim_matrix_all.csv")
	sim.mat <- read.csv(file=tab.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE, row.names=1)
	
	# look for substituted chars
	cat("..Dealing with the substituted characters\n")
	for(i in 1:length(subs.chars))
	{	old.char <- names(subs.chars)[i]
		if(old.char %in% rownames(sim.mat))
		{	cat("......First character:",old.char,"\n")
			
			new.char <- subs.chars[i]
			if(new.char %in% colnames(sim.mat))
			{	cat("......Second character:",new.char,"\n")
				
				self.sim <- sim.mat[old.char,new.char]
				cat("........Self-similarity:",self.sim,"\n")
				#
				ba1 <- which.max(sim.mat[old.char, -which(colnames(sim.mat)==new.char)])
				best.alter1 <- sim.mat[old.char,ba1]
				cat("........Alter similarity 1: ",best.alter1," (",colnames(sim.mat)[ba1],")\n",sep="")
				#
				ba2 <- which.max(sim.mat[-which(colnames(sim.mat)==old.char), new.char])
				best.alter2 <- sim.mat[ba2,new.char]
				cat("........Alter similarity 2: ",best.alter2," (",rownames(sim.mat)[ba2],")\n",sep="")
				#
				cat("......>> Self",if(self.sim>best.alter1 && self.sim>best.alter2) "BETTER" else "worse","than both alters\n")
			}
			else
				cat("......Could not find second character \"",new.char,"\"\n",sep="")
		}
		else
			cat("..Could not find first character \"",old.char,"\"\n",sep="")
	}
	cat("\n")
	
	# look for merged chars
	cat("..Dealing with the merged characters\n")
	for(i in 1:nrow(merged.chars))
	{	merged.char <- rownames(merged.chars)[i]
		if(merged.char %in% colnames(sim.mat))
		{	cat("......Merged character:",merged.char,"\n")
			
			orig.chars <- merged.chars[i,]
			for(j in 1:length(orig.chars))
			{	orig.char <- orig.chars[j]
				if(orig.char %in% rownames(sim.mat))
				{	cat("......Original character:",orig.char,"\n")
					
					self.sim <- sim.mat[orig.char,merged.char]
					cat("........Self-similarity:",self.sim,"\n")
					#
					ba1 <- which.max(sim.mat[-which(rownames(sim.mat)==orig.char), merged.char])
					best.alter1 <- sim.mat[ba1,merged.char]
					cat("........Alter similarity 1: ",best.alter1," (",rownames(sim.mat)[ba1],")\n",sep="")
					#
					ba2 <- which.max(sim.mat[orig.char, -which(colnames(sim.mat)==merged.char)])
					best.alter2 <- sim.mat[orig.char,ba2]
					cat("........Alter similarity 2: ",best.alter2," (",colnames(sim.mat)[ba2],")\n",sep="")
					#
					cat("......>> Self",if(self.sim>best.alter1 && self.sim>best.alter2) "BETTER" else "worse","than both alters\n")
				}
				else
					cat("......Could not find original character \"",orig.char,"\"\n",sep="")
			}
		}
		else
			cat("..Could not find merged character \"",merged.char,"\"\n",sep="")
	}
	cat("\n\n")
}




###############################################################################
# second approach: working directly with the networks
for(NARRATIVE_PART in c(2,5))
{	cat(">>>>>>>>>> Processing narrative part = ",NARRATIVE_PART," <<<<<<<<<<\n",sep="")
	
	# load the static graphs and rank the characters by importance
	source("src/common/load_static_nets.R")
	
	# loop over character sets
	for(CHARSET in c("named","common","top"))
	{	cat("....Processing character set ",CHARSET,"\n",sep="")
		
		# look for substituted chars
		cat("......Dealing with the substituted characters\n")
		for(i in 1:length(subs.chars))
		{	g1 <- gs[["novels"]]
			g2 <- gs[["tvshow"]]
			
			old.char <- names(subs.chars)[i]
			if(old.char %in% V(g1)$name)
			{	cat("..........First character:",old.char,"\n")
				
				new.char <- subs.chars[i]
				if(new.char %in% V(g2)$name)
				{	cat("..........Second character:",new.char,"\n")
					
					# prepare the graphs
					if(CHARSET=="top")
					{	# remove non-important chars (except those compared)
						names <- union(ranked.chars[1:TOP_CHAR_NBR],c(old.char,new.char))
						idx1 <- which(!(V(g1)$name %in% names))
						g1 <- delete_vertices(g1,idx1)
						idx2 <- which(!(V(g2)$name %in% names))
						g2 <- delete_vertices(g2,idx2)
					}
					# or focus on characters common to both networks
					else if(CHARSET=="common")
					{	# remove chars that are not common (except those compared)
						names <- union(intersect(V(g1)$name,V(g2)$name),c(old.char,new.char))
						idx1 <- which(!(V(g1)$name %in% names))
						g1 <- delete_vertices(g1,idx1)
						idx2 <- which(!(V(g2)$name %in% names))
						g2 <- delete_vertices(g2,idx2)
					}
					# complete the graphs to have the same characters in both
					# get the name lists
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
					
					# compute and normalize adjacency matrices
					a1 <- as_adjacency_matrix(graph=g1, type="both", attr="weight", sparse=FALSE)
					a1 <- t(apply(a1, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
					a2 <- as_adjacency_matrix(graph=g2, type="both", attr="weight", sparse=FALSE)
					a2 <- t(apply(a2, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
					
					# compute jaccard (weighted) similarity
					idx1 <- which(names==old.char)
					idx2 <- which(names==new.char)
					# similarity for the first character
					sims.v1 <- rep(NA,length(names))
					v1 <- idx1
					for(v2 in 1:nrow(a2)) 
					{	w.min <- pmin(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						w.max <- pmax(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						if(sum(w.max)==0)
							sim <- 0
						else
							sim <- sum(w.min)/sum(w.max)
						#print(sim)
						sims.v1[v2] <- sim
					}
					# similarity for the second character
					sims.v2 <- rep(NA,length(names))
					v2 <- idx2
					for(v1 in 1:nrow(a1)) 
					{	w.min <- pmin(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						w.max <- pmax(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						if(sum(w.max)==0)
							sim <- 0
						else
							sim <- sum(w.min)/sum(w.max)
						#print(sim)
						sims.v2[v1] <- sim
					}
					
					self.sim <- sims.v1[idx2]
					cat("............Self-similarity:",sims.v1[idx2]," () ",sims.v2[idx1],"\n")
					#
					ba1 <- which.max(sims.v1[-idx2])
					best.alter1 <- sims.v1[ba1]
					cat("............Alter similarity 1: ",best.alter1," (",names[ba1],")\n",sep="")
					#
					ba2 <- which.max(sims.v1[-idx2])
					best.alter2 <- sims.v2[ba2]
					cat("............Alter similarity 2: ",best.alter2," (",names[ba2],")\n",sep="")
					#
					cat("..........>> Self",if(self.sim>best.alter1 && self.sim>best.alter2) "BETTER" else "worse","than both alters\n")
				}
				else
					cat("..........Could not find second character \"",new.char,"\"\n",sep="")
			}
			else
				cat("......Could not find first character \"",old.char,"\"\n",sep="")
		}
		cat("\n")
		
		# look for merged chars
		# TODO could not test this part: too many characters are missing from the networks, esp. TV show S5 (they appear in later seasons)
		cat("......Dealing with the merged characters\n")
		for(i in 1:nrow(merged.chars))
		{	g1 <- gs[["novels"]]
			g2 <- gs[["tvshow"]]
			
			merged.char <- rownames(merged.chars)[i]
			if(merged.char %in% V(g2)$name)
			{	cat("......Merged character:",merged.char,"\n")
				
				orig.chars <- merged.chars[i,]
				if(all(orig.chars %in% V(g1)$name))
				{	cat("..........Original characters:",paste0(orig.chars,collapse=","),"\n")
					
					# prepare the graphs
					g1 <- gs[["novels"]]
					g2 <- gs[["tvshow"]]
					# merge the considered characters in the first graph
					idx <- which(V(g1)$name %in% orig.chars)
					map <- 1:gorder(g1)
					map[idx] <- min(idx)
					map[map>idx] <- map[map>idx] - length(idx)
					orig.char <- V(g1)$name[idx[1]]
					g1 <- contract(graph=g1, mapping=map)
					g1 <- simplify(graph=g1, edge.attr.comb=list(weight="mean", "ignore"))
					# handle top char nets
					if(CHARSET=="top")
					{	# remove non-important chars (except those compared)
						names <- union(ranked.chars[1:TOP_CHAR_NBR],c(merged.char,orig.char))
						idx1 <- which(!(V(g1)$name %in% names))
						g1 <- delete_vertices(g1,idx1)
						idx2 <- which(!(V(g2)$name %in% names))
						g2 <- delete_vertices(g2,idx2)
					}
					# or focus on characters common to both networks
					else if(CHARSET=="common")
					{	# remove chars that are not common (except those compared)
						names <- union(intersect(V(g1)$name,V(g2)$name),c(merged.char,orig.char))
						idx1 <- which(!(V(g1)$name %in% names))
						g1 <- delete_vertices(g1,idx1)
						idx2 <- which(!(V(g2)$name %in% names))
						g2 <- delete_vertices(g2,idx2)
					}
					# complete the graphs to have the same characters in both
					# get the name lists
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
					
					# compute and normalize adjacency matrices
					a1 <- as_adjacency_matrix(graph=g1, type="both", attr="weight", sparse=FALSE)
					a1 <- t(apply(a1, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
					a2 <- as_adjacency_matrix(graph=g2, type="both", attr="weight", sparse=FALSE)
					a2 <- t(apply(a2, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
					
					# compute jaccard (weighted) similarity
					idx1 <- which(names==merged.char)
					idx2 <- which(names==orig.char)
					# similarity for the first character
					sims.v1 <- rep(NA,length(names))
					v1 <- idx1
					for(v2 in 1:nrow(a2)) 
					{	w.min <- pmin(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						w.max <- pmax(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						if(sum(w.max)==0)
							sim <- 0
						else
							sim <- sum(w.min)/sum(w.max)
						#print(sim)
						sims.v1[v2] <- sim
					}
					# similarity for the second character
					sims.v2 <- rep(NA,length(names))
					v2 <- idx2
					for(v1 in 1:nrow(a1)) 
					{	w.min <- pmin(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						w.max <- pmax(a1[v1,-c(idx1,idx2)], a2[v2,-c(idx1,idx2)])
						if(sum(w.max)==0)
							sim <- 0
						else
							sim <- sum(w.min)/sum(w.max)
						#print(sim)
						sims.v2[v1] <- sim
					}
					
					self.sim <- sims.v1[idx2]
					cat("............Self-similarity:",self.sim,"\n")
					#
					ba1 <- which.max(sims.v1[-idx2])
					best.alter1 <- sims.v1[ba1]
					cat("............Alter similarity 1: ",best.alter1," (",names[ba1],")\n",sep="")
					#
					ba2 <- which.max(sims.v1[-idx2])
					best.alter2 <- sims.v2[ba2]
					cat("............Alter similarity 2: ",best.alter2," (",names[ba2],")\n",sep="")
					#
					cat("..........>> Self",if(self.sim>best.alter1 && self.sim>best.alter2) "BETTER" else "worse","than both alters\n")
				}
				else
					cat("..........Could not find at least one of the original characters\n",sep="")
			}
			else
				cat("......Could not find merged character \"",merged.char,"\"\n",sep="")
		}
		cat("\n\n")
	}
}
