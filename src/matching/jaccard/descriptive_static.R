# Computes the similarity between a character and its counterpart in a different
# network.
# 
# Author: Vincent Labatut
# 08/2023
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/jaccard/descriptive_static.R")
###############################################################################
library("igraph")
library("viridis")
library("plot.matrix")




###############################################################################
# processing parameters
COMMON_CHARS_ONLY <- TRUE	# all named characters, or only those common to both compared graphs
MEAS <- "jaccard"			# no alternative for now
TOP_CHAR_NBR <- 20			# number of important characters




###############################################################################
# output folder
out.folder <- file.path("out","matching",MEAS)
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)

{	if(COMMON_CHARS_ONLY)
		mode.folder <- "common"
	else
		mode.folder <- "named"
}




###############################################################################
# only take the first two narrative units (whole narrative not supported here)
NARRATIVE_PART <- 2
# load the static graphs and rank the characters by importance
source("src/common/load_static_nets.R")




###############################################################################
# start matching, looping over pairs of networks
for(i in 1:(length(gs)-1))
{	cat("..Processing first network ",g.names[i],"\n",sep="")
	
	for(j in (i+1):length(gs))
	{	cat("....Processing second network ",g.names[j],"\n",sep="")
		g1 <- gs[[i]]
		g2 <- gs[[j]]
		
		# init local folder
		comp.name <- paste0(g.names[i], "_vs_", g.names[j])
		local.folder <- file.path(out.folder, mode.folder, comp.name)
		dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
		
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
		
# older, much slower version		
#		# build the vertex similarity matrix
#		sim.mat <- outer(X=1:gorder(g1), Y=1:gorder(g2), FUN=function(vv1,vv2)
#		{	sapply(1:length(vv1),function(idx)
#			{	v1 <- vv1[idx]
#				v2 <- vv2[idx]
#				#print(v1);print(v2)
#				n1 <- neighbors(graph=g1,v=v1,mode="all")$name
#				w1 <- E(g1)[V(g1)[v1] %--% n1]$weight
#				w1 <- w1 / sum(w1)
#				n2 <- neighbors(graph=g2,v=v2,mode="all")$name
#				w2 <- E(g2)[V(g2)[v2] %--% n2]$weight
#				w2 <- w2 / sum(w2)
#				if(MEAS=="jaccard")
#				{	if(length(n1)==0 || length(n2)==0)
#						sim <- 0
#					else
#					{	all <- union(n1,n2)
#						j1 <- rep(0,length(all))
#						j2 <- rep(0,length(all))
#						j1[match(n1,all)] <- w1
#						j2[match(n2,all)] <- w2
#						j.min <- apply(cbind(j1,j2), 1, min)
#						j.max <- apply(cbind(j1,j2), 1, max)
#						sim <- sum(j.min)/sum(j.max)
#					}
#				}
#				#print(sim)
#				return(sim)
#			})
#		})
# alt version (as slow)
#		sim.mat <- matrix(NA, nrow=gorder(g1), ncol=gorder(g2))
#		for(v1 in 1:gorder(g1))
#		{	print(v1)
#			for(v2 in 1:gorder(g2)) 
#			{	n1 <- neighbors(graph=g1,v=v1,mode="all")$name
#				w1 <- E(g1)[V(g1)[v1] %--% n1]$weight
#				w1 <- w1 / sum(w1)
#				n2 <- neighbors(graph=g2,v=v2,mode="all")$name
#				w2 <- E(g2)[V(g2)[v2] %--% n2]$weight
#				w2 <- w2 / sum(w2)
#				if(MEAS=="jaccard")
#				{	if(length(n1)==0 || length(n2)==0)
#						sim <- 0
#					else
#					{	all <- union(n1,n2)
#						j1 <- rep(0,length(all))
#						j2 <- rep(0,length(all))
#						j1[match(n1,all)] <- w1
#						j2[match(n2,all)] <- w2
#						j.min <- apply(cbind(j1,j2), 1, min)
#						j.max <- apply(cbind(j1,j2), 1, max)
#						sim <- sum(j.min)/sum(j.max)
#					}
#				}
#				#print(sim)
#				sim.mat[v1,v2] <- sim
#			}
#		}
		
		# compute and normalize adjacency matrices
		a1 <- as_adjacency_matrix(graph=g1, type="both", attr="weight", sparse=FALSE)
		a1 <- t(apply(a1, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
		a2 <- as_adjacency_matrix(graph=g2, type="both", attr="weight", sparse=FALSE)
		a2 <- t(apply(a2, 1, function(row) if(sum(row)==0) rep(0,length(row)) else row/sum(row)))
		names <- V(g1)$name
		
		if(MEAS=="jaccard")
		{	# compute jaccard (weighted) similarity
#			sim.mat <- outer(X=1:nrow(a1), Y=1:nrow(a2), FUN=function(vv1,vv2)
#			{	sapply(1:length(vv1),function(idx)
#				{	#print(v1)
#					v1 <- vv1[idx]
#					v2 <- vv2[idx]
#					w.min <- pmin(a1[v1,], a2[v2,])
#					w.max <- pmax(a1[v1,], a2[v2,])
#					sum(w.min)/sum(w.max)
#				})
#			})
			# alt version (seems faster)
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
		
		# plot matrix
		plot.file <- file.path(local.folder,"sim_matrix_all")
		pdf(paste0(plot.file,".pdf"), bg="white", width=30, height=30)
			plot(sim.mat[idx,idx], border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=comp.name, cex.axis=0.2)
		dev.off()
		# plot only top characters
		plot.file <- file.path(local.folder,paste0("sim_matrix_top",TOP_CHAR_NBR))
		pdf(paste0(plot.file,".pdf"), bg="white")
			#par(mar=c(4,4,0,0)+0.1)	# remove the title space Bottom Left Top Right
			plot(sim.mat[idx[1:TOP_CHAR_NBR],idx[1:TOP_CHAR_NBR]], border=NA, col=viridis, las=2, xlab=NA, ylab=NA, main=comp.name, cex.axis=0.5)
		dev.off()
		
		# compute some form of performance by considering the most similar alters vs. self
		sim.self <- diag(sim.mat)
		tmp <- sim.mat; diag(tmp) <- 0
		sim.alter1 <- apply(tmp, 1, max)
		sim.alter2 <- apply(tmp, 2, max)
		sim.alter <- pmax(sim.alter1, sim.alter2)
		d1 <- degree(g1,mode="all")
		d2 <- degree(g1,mode="all")
		acc1 <- length(which(sim.self>sim.alter2))/length(d1>0)
		acc2 <- length(which(sim.self>sim.alter1))/length(d2>0)
		acc <- length(which(sim.self>sim.alter))/length(sim.self)
		perf.tab <- c(acc1,acc2,acc)
		# only top characters
		idx <- idx[1:TOP_CHAR_NBR]
		acc1 <- length(which(sim.self[idx]>sim.alter2[idx]))/length(d1[idx]>0)
		acc2 <- length(which(sim.self[idx]>sim.alter1[idx]))/length(d2[idx]>0)
		acc <- length(which(sim.self[idx]>sim.alter[idx]))/length(sim.self[idx])
		perf.tab <- rbind(perf.tab, c(acc1,acc2,acc))
		rownames(perf.tab) <- c("All","Top-20")
		colnames(perf.tab) <- c(comp.name,paste0(g.names[j], "_vs_", g.names[i]),"overall")
		write.csv(x=perf.tab, file=file.path(local.folder,"sim_perf.csv"), row.names=FALSE, fileEncoding="UTF-8")
		cat("Performance when matching to the most similar character:\n",sep="");print(perf.tab)
		
		# plot self vs. best alter
		imp.vals <- imp.moy[ranked.names]
		probs <- c(0.5,0.75,0.9,1.0)
		fprobs <- sprintf("%1.2f",probs*100)
		quant <- quantile(imp.vals, probs=probs)
		pal <- viridis(4, alpha=0.75)
		cols <- pal[as.numeric(cut(imp.vals,breaks=c(0,quant),include.lowest=TRUE))]
		# produce plots
		for(s in c(NA,1,2,3,4))
		{	if(is.na(s))
			{	idx <- TRUE
				plot.file <- file.path(local.folder,paste0("sim_self_vs_bestalter_all"))
			}
			else
			{	idx <- cols==pal[s]
				plot.file <- file.path(local.folder,paste0("sim_self_vs_bestalter_s",fprobs[s]))
			}
			pdf(paste0(plot.file,".pdf"), bg="white")
				plot(
					NULL, 
					main=comp.name, xlab="Self-similarity", ylab="Best alter-similarity",
					xlim=0:1, ylim=0:1
				)
				abline(a=0,b=1,col="BLACK",lty=3)
				points(
					x=sim.self[idx], y=sim.alter[idx],
					col=cols[idx], pch=16
				)
				legend(
					"bottomright",
					fill=pal,
					legend=paste0(fprobs,"%"),
					title="Vertices"
				)
			dev.off()
		}
	}
}
