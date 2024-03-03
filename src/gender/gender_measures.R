# Functions allowing to compute the gender-related measures.
# 
# Author: Vincent Labatut
# 02/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/gender/gender_measures.R")
###############################################################################
library("igraph")




###############################################################################
# Computes all the gender-related topological measures, for the specified graph.
#
# g: graph of interest.
#
# returns: a named list containing the computed values.
###############################################################################
compute.gender.measures <- function(g)
{	sexes <- c("Male", "Female", "Mixed", "Unknown")
	res <- list()
	
	# vertex counts
	if(gorder(g)>0)
	{	cnt.m <- length(which(V(g)$sex=="Male"))
		cnt.f <- length(which(V(g)$sex=="Female"))
		cnt.o <- gorder(g) - cnt.m - cnt.f
		tab <- c(cnt.m,cnt.f,cnt.o)
		per.m <- cnt.m/gorder(g)*100
		per.f <- cnt.f/gorder(g)*100
		per.o <- cnt.o/gorder(g)*100
		tab <- cbind(tab, c(per.m,per.f,per.o))
		per.m <- cnt.m/(cnt.m+cnt.f)*100
		per.f <- cnt.f/(cnt.m+cnt.f)*100
		tab <- cbind(tab, c(per.m,per.f,NA))
	}
	else
		tab <- matrix(0, nrow=3, ncol=3)
	rownames(tab) <- c("Male","Female","Other")
	colnames(tab) <- c("Count","OverallProportion","FM_Proportion")
	res[["vertices"]] <- tab
	
	# edge counts
	if(gsize(g)>0)
	{	el <- as_edgelist(graph=g, names=FALSE)
		el.sex <- cbind(V(g)[el[,1]]$sex, V(g)[el[,2]]$sex)
		el.str <- t(apply(el.sex, 1, sort))
		# filter unknown sex and convert mixed sex
		idx.mxd <- which(el.str[,1]=="Female" & el.str[,2]=="Mixed")
		el.str[idx.mxd,2] <- rep("Male",length(idx.mxd))
		idx.mxd <- which(el.str[,1]=="Male" & el.str[,2]=="Mixed")
		el.str[idx.mxd,1] <- rep("Female",length(idx.mxd))
		el.str[idx.mxd,2] <- rep("Male",length(idx.mxd))
		idx.mxd <- which(el.str[,1]=="Mixed" & el.str[,2]=="Mixed")
		el.str[idx.mxd,1] <- rep("Female",length(idx.mxd))
		el.str[idx.mxd,2] <- rep("Male",length(idx.mxd))
		idx.ukn <- which(el.str[,1]=="Unknown" | el.str[,2]=="Unknown")
		if(length(idx.ukn)>0)
			el.str <- el.str[-idx.ukn,]
		levs <- sort(c("Female","Male"))
		tt <- table(factor(el.str[,1], levs), factor(el.str[,2], levs))
		tab <- c(t(tt)[c(1,2,4)],NA)
		tt <- tt/sum(tt)*100
		tab <- cbind(tab, c(t(tt)[c(1,2,4)],NA))
		ratio <- sum(tt["Female",])/sum(tt[,"Male"])	# nbr of edges connected to at least one female /to at least one male 
		tab <- cbind(tab, c(NA,NA,NA,ratio))
		# homophily
		gs <- delete_vertices(g, V(g)$sex!="Female" & V(g)$sex!="Male")
		ass <- assortativity_nominal(graph=gs, types=factor(V(gs)$sex), directed=FALSE)
		tab <- cbind(tab, c(NA,NA,NA,ass))
	}
	else
	{	tab <- matrix(NA, nrow=4, ncol=4)
		tab[,1] <- tab[,2] <- c(0,0,0,NA)
	}
	# finalize table
	rownames(tab) <- c("F-F","F-M","M-M","All")
	colnames(tab) <- c("Count","Proportion","F-/M- Ratio","Homophily")
	res[["edges"]] <- tab
	
	# degree
	vals <- degree(graph=g, mode="all", normalized=FALSE)
	cnt <- sapply(sexes, function(s) sum(vals[V(g)$sex==s]))
	tt <- sum(cnt)
	if(tt==0) tt <- 1
	tab <- cbind(cnt,cnt/tt*100)
	ratio <- cnt["Female"]/cnt["Male"]
	tab <- rbind(tab, c(NA,NA))
	tab <- cbind(tab, c(rep(NA,length(sexes)), ratio))
	rownames(tab) <- c(sexes,"All")
	colnames(tab) <- c("Total","Proportion","GenderDegreeRatio")
	res[["degree"]] <- tab
	
	# strength
	vals <- strength(graph=g, mode="all", weights=E(g)$weight)
	cnt <- sapply(sexes, function(s) sum(vals[V(g)$sex==s]))
	tt <- sum(cnt)
	if(tt==0) tt <- 1
	tab <- cbind(cnt,cnt/tt*100)
	ratio <- cnt["Female"]/cnt["Male"]
	tab <- rbind(tab, c(NA,NA))
	tab <- cbind(tab, c(rep(NA,length(sexes)), ratio))
	rownames(tab) <- c(sexes,"All")
	colnames(tab) <- c("Total","Proportion","GenderStrengthRatio")
	res[["strength"]] <- tab

	# density
	if(gorder(g)>0)
	{	dens.all <- edge_density(g)
		gm <- delete_vertices(g, V(g)$sex!="Male")
		dens.mal <- edge_density(gm)
		gf <- delete_vertices(g, V(g)$sex!="Female")
		dens.fem <- edge_density(gf)
		tab <- matrix(c(dens.mal, dens.fem, dens.all),ncol=1)
	}
	else
		tab <- matrix(0,nrow=3,ncol=1)
	colnames(tab) <- c("Density")
	rownames(tab) <- c("Male", "Female", "All")
	res[["density"]] <- tab
	
	# triangle counts
	gs <- delete_vertices(g, V(g)$sex!="Female" & V(g)$sex!="Male")
	tris <- triangles(graph=gs)$sex
	col.tris <- sapply(1:(length(tris)/3), function(t) paste(sort(tris[(t*3-2):(t*3)]),collapse="-"))
	levs <- sort(c("Female-Female-Female","Female-Female-Male","Female-Male-Male","Male-Male-Male"))
	tt <- table(factor(col.tris,levs))
	tab <- c(tt)
	ss <- sum(tt)
	if(ss==0) ss <- 1
	tt <- tt/ss*100
	tab <- cbind(tab, c(tt))
	colnames(tab) <- c("Count", "FM_Proportion")
	res[["triangles"]] <- tab
	
#	res[["scalars"]] <- vals
	return(res)
}
