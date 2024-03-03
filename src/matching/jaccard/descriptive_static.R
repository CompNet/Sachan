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

source("src/common/colors.R")




###############################################################################
# processing parameters
NARRATIVE_PART <- 2			# take the first two (2) or five (5) narrative units
CHARSET <- "common"			# all named characters (named), or only those common to both compared graphs (common), or the 20 most important (top)
MEAS <- "jaccard"			# no alternative for now
TOP_CHAR_NBR <- 20			# number of important characters (fixed)
PLOT_CHAR_NAMES <- FALSE	# whether to plot the character names in the larger plots




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
out.folder <- file.path(out.folder, MEAS)
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)

{	if(CHARSET=="top")
		mode.folder <- paste0(CHARSET,TOP_CHAR_NBR)
	else
		mode.folder <- CHARSET
}




###############################################################################
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
		#comp.title <- paste0(narr.names[g.names[i]], " vs. ", narr.names[g.names[j]])
		comp.title <- bquote(bolditalic(.(narr.names[g.names[i]]))~bold(" vs. ")~bolditalic(.(narr.names[g.names[j]])))
		local.folder <- file.path(out.folder, mode.folder, comp.name)
		dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
		
		# focus on the most important characters
		if(CHARSET=="top")
		{	# remove non-important chars
			names <- ranked.chars[1:TOP_CHAR_NBR]
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
		# or focus on characters common to both networks
		else if(CHARSET=="common")
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
		
		# init matching performance matrix
		rn <- c("All","Top-20")
		cn <- c(comp.name,paste0(g.names[j], "_vs_", g.names[i]),"overall")
		perf.tab <- matrix(NA, nrow=length(rn), ncol=length(cn))
		rownames(perf.tab) <- rn
		colnames(perf.tab) <- cn
		
		# compute and normalize adjacency matrices
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
			write.csv(x=sim.mat, file=file.path(local.folder,"sim_matrix_all.csv"), row.names=TRUE, fileEncoding="UTF-8")
		}
		
		# plot similarity matrix
		ranked.names <- setdiff(ranked.chars, setdiff(ranked.chars, names))
		idx <- match(ranked.names, names)
		plot.file <- file.path(local.folder,"sim_matrix_all")
		if(PLOT_CHAR_NAMES)
		{	pdf(paste0(plot.file,".pdf"), width=30, height=30)	# bg="white"
				par(mar=c(5,4,4,2)+0.1)	# margins Bottom Left Top Right
				plot(
					sim.mat[idx,idx], 
					border=NA, col=viridis, breaks=seq(0.0,1,0.1), 
					las=2, 
					xlab=bquote(italic(.(narr.names[g.names[i]]))), ylab=bquote(italic(.(narr.names[g.names[j]]))), main=NA, 
					cex.axis=0.2,
					key=NULL
				)
				title(comp.title,  line=1)
			dev.off()
		}
		else
		{	pdf(paste0(plot.file,".pdf"), width=7, height=7)	# bg="white"
				par(mar=c(3,2,2,0.5)+0.1)	# margins Bottom Left Top Right
				plot(
					sim.mat[idx,idx], 
					border=NA, col=viridis, breaks=seq(0.0,1,0.1), 
					las=2, 
					xlab=bquote(italic(.(narr.names[g.names[i]]))), ylab=bquote(italic(.(narr.names[g.names[j]]))), main=NA, 
					axis.col=NULL, axis.row=NULL, mgp=c(1,1,0),
					key=NULL
				)
				title(comp.title,  line=1)
			dev.off()
		}
		# plot only top characters
		plot.file <- file.path(local.folder,paste0("sim_matrix_top",TOP_CHAR_NBR))
		pdf(paste0(plot.file,".pdf"))	# bg="white"
			par(mar=c(5.5,4.75,4.5,2)+0.1)	# margins Bottom Left Top Right
			plot(
				sim.mat[idx[1:TOP_CHAR_NBR],idx[1:TOP_CHAR_NBR]], 
				border=NA, col=viridis, breaks=seq(0.0,1,0.1), 
				las=2, 
				xlab=NA, ylab=NA, main=comp.title, 
				cex.axis=0.5, fmt.key="%.2f"
			)
		dev.off()
		
		# compute some sort of performance by considering the most similar alters vs. self
		sim.self <- diag(sim.mat)
		tmp <- sim.mat; diag(tmp) <- 0
		sim.alter1 <- apply(tmp, 1, max)
		sim.alter2 <- apply(tmp, 2, max)
		sim.alter <- pmax(sim.alter1, sim.alter2)
		d1 <- degree(g1,mode="all")
		d2 <- degree(g2,mode="all")
		acc1 <- length(which(sim.self>sim.alter2))/length(d1>0)
		acc2 <- length(which(sim.self>sim.alter1))/length(d2>0)
		acc <- length(which(sim.self>sim.alter))/length(sim.self)
		cat("  Number of characters used to compute the perf:",length(sim.self),"\n")
		perf.tab["All",] <- c(acc1,acc2,acc)
		# focus only on most important characters
		idx <- idx[1:TOP_CHAR_NBR]
		acc1 <- length(which(sim.self[idx]>sim.alter2[idx]))/length(d1[idx]>0)
		acc2 <- length(which(sim.self[idx]>sim.alter1[idx]))/length(d2[idx]>0)
		acc <- length(which(sim.self[idx]>sim.alter[idx]))/length(sim.self[idx])
		perf.tab["Top-20",] <- c(acc1,acc2,acc)
		# record
		cat("  Number of characters used to compute the top-20 perf:",length(idx),"\n")
		write.csv(x=perf.tab, file=file.path(local.folder,"sim_perf.csv"), row.names=TRUE, fileEncoding="UTF-8")
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
			pdf(paste0(plot.file,".pdf"))	# bg="white"
				plot(
					NULL, 
					main=comp.title, xlab="Self-similarity", ylab="Best alter-similarity",
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
					title="Vertices",
					bg="#FFFFFFCC"
				)
			dev.off()
		}
		
		# define a table to store correlation values
		rn <- c("Self-sim_vs_Imprt","Sim-diff_vs_Imprt")
		cn <- c("PearsonCoef","PearsonPval","SpearmanCoef","SpearmanPval","KendallCoef","KendallPval")
		corr.tab <- matrix(NA,nrow=length(rn), ncol=length(cn))
		colnames(corr.tab) <- cn
		rownames(corr.tab) <- rn
		
		# set up colors for next plots
		transp <- 25	# transparency level
		pal <- get.palette(2)
		cols <- rep(make.color.transparent(pal[2],transp), gorder(g1))
		cols[which(V(g1)$name %in% ranked.chars[1:TOP_CHAR_NBR])] <- make.color.transparent(pal[1],transp)
		
		# plot self-similarity vs. character importance
		imp <- char.importance[match(V(g1)$name,char.importance[,"Name"]),"Mean"]
		yvals <- sim.self
		res <- cor.test(x=imp, y=yvals, method="pearson")
		corr.tab["Self-sim_vs_Imprt","PearsonCoef"] <- res$estimate
		corr.tab["Self-sim_vs_Imprt","PearsonPval"] <- res$p.value
		res <- cor.test(x=imp, y=yvals, method="spearman")
		corr.tab["Self-sim_vs_Imprt","SpearmanCoef"] <- res$estimate
		corr.tab["Self-sim_vs_Imprt","SpearmanPval"] <- res$p.value
		res <- cor.test(x=imp, y=yvals, method="kendall")
		corr.tab["Self-sim_vs_Imprt","KendallCoef"] <- res$estimate
		corr.tab["Self-sim_vs_Imprt","KendallPval"] <- res$p.value
		plot.file <- file.path(local.folder,paste0("similarity-self_vs_importance"))
		pdf(paste0(plot.file,".pdf"), width=7, height=7)	# bg="white"
			par(mar=c(5, 4, 4, 2)+0.1)	# margins Bottom Left Top Right
			plot(
				NULL,
				log="x", col=cols, 
				main=comp.title, xlab="Importance", ylab="Self-similarity",
				xlim=range(imp), ylim=range(yvals)
			)
			points(
				imp, yvals,
				pch=16, col=cols, 
			)
			legend(
				x="topright",
				title="Characters",
				legend=c(paste0("Top-",TOP_CHAR_NBR),"Others"),
				fill=pal,
				bg="#FFFFFFCC"
			)
		dev.off()
		
		# plot self-similarity-best alter vs. character importance
		imp <- char.importance[match(V(g1)$name,char.importance[,"Name"]),"Mean"]
		yvals <- sim.self-sim.alter
		res <- cor.test(x=imp, y=yvals, method="pearson")
		corr.tab["Sim-diff_vs_Imprt","PearsonCoef"] <- res$estimate
		corr.tab["Sim-diff_vs_Imprt","PearsonPval"] <- res$p.value
		res <- cor.test(x=imp, y=yvals, method="spearman")
		corr.tab["Sim-diff_vs_Imprt","SpearmanCoef"] <- res$estimate
		corr.tab["Sim-diff_vs_Imprt","SpearmanPval"] <- res$p.value
		res <- cor.test(x=imp, y=yvals, method="kendall")
		corr.tab["Sim-diff_vs_Imprt","KendallCoef"] <- res$estimate
		corr.tab["Sim-diff_vs_Imprt","KendallPval"] <- res$p.value
		plot.file <- file.path(local.folder,paste0("similarity-diff_vs_importance"))
		pdf(paste0(plot.file,".pdf"), width=7, height=7)	# bg="white"
			par(mar=c(5, 4, 4, 2)+0.1)	# margins Bottom Left Top Right
			plot(
				NULL, 
				log="x", 
				main=comp.title, xlab="Importance", ylab="Difference between self and best alter similarities",
				xlim=range(imp), ylim=range(yvals)
			)
			abline(h=0, lty=2)
			points(
				imp, yvals,
				pch=16, col=cols, 
			)
			legend(
				x="bottomright",
				title="Characters",
				legend=c(paste0("Top-",TOP_CHAR_NBR),"Others"),
				fill=pal,
				bg="#FFFFFFCC"
			)
		dev.off()
		
		# record correlation matrix
		cat("Correlation matrix:\n"); print(corr.tab)
		tab.file <- file.path(local.folder,"sim-imprt_corr.csv")
		write.csv(x=corr.tab, file=tab.file, row.names=TRUE, fileEncoding="UTF-8")
		
		# note: correlation test
		# 	h_0: no linear relationship between the two variables
		#	p<alpha => reject the null hypothesis, i.e. there is a relationship
	}
}
