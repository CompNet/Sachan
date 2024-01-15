# Compute character frequency (number of occurrences) to use as importance.
# Record the corresponding files for later use, if they do not already exist.
# 
# Author: Vincent Labatut
# 01/2024
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/common/char_importance.R")
###############################################################################




###############################################################################
# check whether the files already exist
narr.units <- c(0,2,5)
out.files <- paste0(file.path("in","ranked_importance_S"), narr.units, ".csv")
if(all(file.exists(out.files)))
	cat("All the files listing important characters already exist\n")

	
	
	
###############################################################################
# loop over numbers of seasons/books
if(!all(file.exists(out.files)))
{	# read instant dynamic networks for novels
	cat("Reading novel networks\n",sep="")
	gs.nv <- list()
	names.nv <- c()
	files <- sort(list.files(path=file.path("in","novels","instant"), pattern=".+\\.graphml", full.names=TRUE))
	for(i in 1:length(files))
	{	cat("..Loading file \"",files[i],"\"\n",sep="")
		
		# load graph
		g.nv <- read.graph(files[i], format="graphml")
		g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)			# keep only named characters
		g.nv$file <- files[i]
		
		# retrieve character names
		names.nv <- sort(union(names.nv, V(g.nv)$name))
		
		gs.nv <- c(gs.nv, list(g.nv))
	}
	cat("Loaded a total of ",length(gs.nv)," novel networks\n",sep="")
	
	# read instant dynamic networks for comics
	cat("Reading comics networks\n",sep="")
	gs.cx <- list()
	names.cx <- c()
	files <- sort(list.files(path=file.path("in","comics","instant","scene"), pattern=".+\\.graphml", full.names=TRUE))
	for(i in 1:length(files))
	{	cat("..Loading file \"",files[i],"\"\n",sep="")
		
		# load graph
		g.cx <- read.graph(files[i], format="graphml")
		g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)	# keep only named characters
		if(gsize(g.cx)>0)
			E(g.cx)$weight <- E(g.cx)$weight/max(E(g.cx)$weight)	# normalize weights
		g.cx$file <- files[i]
		
		# retrieve character names
		names.cx <- sort(union(names.cx, V(g.cx)$name))
		
		gs.cx <- c(gs.cx, list(g.cx))
	}
	cat("Loaded a total of ",length(gs.cx)," comic networks\n",sep="")
	
	# read instant dynamic networks for tv show
	cat("Reading TV show networks\n",sep="")
	gs.tv <- list()
	names.tv <- c()
	files <- sort(list.files(path=file.path("in","tvshow","instant","scene"), pattern=".+\\.graphml", full.names=TRUE))
	for(i in 1:length(files))
	{	cat("..Loading file \"",files[i],"\"\n",sep="")
		
		# load graph
		g.tv <- read.graph(files[i], format="graphml")
		if(gorder(g.tv)>0)
			g.tv <- delete_vertices(graph=g.tv, v=!V(g.tv)$named)			# keep only named characters
		if(gsize(g.tv)>0)
			E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)			# normalize weights
		g.tv$file <- files[i]
		
		# retrieve character names
		names.tv <- sort(union(names.tv, V(g.tv)$name))
		
		gs.tv <- c(gs.tv, list(g.tv))
	}
	cat("Loaded a total of ",length(gs.tv)," TV show networks\n",sep="")
				
	for(s in narr.units)
	{	cat("Process narrative unit mode S",s,"\n",sep="")
		
		# parameters
		{	if(s==0)
			{	file.nv <- "in/novels/instant/5.ADwD_72_instant.graphml"
				file.cx <- "in/comics/instant/scene/inst_1437.graphml"
				file.tv <- "in/tvshow/instant/scene/instant_4164.graphml"
			}
			else if(s==2)
			{	file.nv <- "in/novels/instant/2.ACoK_69_instant.graphml"
				file.cx <- "in/comics/instant/scene/inst_1437.graphml"
				file.tv <- "in/tvshow/instant/scene/instant_0753.graphml"
			}
			else if(s==5)
			{	file.nv <- "in/novels/instant/5.ADwD_72_instant.graphml"
				file.cx <- NA
				file.tv <- "in/tvshow/instant/scene/instant_2248.graphml"
			}
		}
		
		# compute character occurrences for novels
		occ.nv <- rep(0, length(names.nv))
		names(occ.nv) <- names.nv
		t <- 1
		cat("Counting occurrences in the novels\n")
		while(t==1 || gs.nv[[t-1]]$file!=file.nv)
		{	#cat("  Processing network \"",gs.nv[[t]]$file,"\"\n",sep="")
			chars <- V(gs.nv[[t]])$name
			occ.nv[chars] <- occ.nv[chars] + 1
			t <- t + 1
		}
		print(occ.nv)
		
		# compute character occurrences for comics
		occ.cx <- rep(0, length(names.cx))
		names(occ.cx) <- names.cx
		if(s<5)
		{	t <- 1
			cat("Counting occurrences in the comics\n")
			while(t==1 || gs.cx[[t-1]]$file!=file.cx)
			{	#cat("  Processing network \"",gs.cx[[t]]$file,"\"\n",sep="")
				chars <- V(gs.cx[[t]])$name
				occ.cx[chars] <- occ.cx[chars] + 1
				t <- t + 1
			}
		}
		print(occ.cx)
		
		# compute character occurrences for tv show
		occ.tv <- rep(0, length(names.tv))
		names(occ.tv) <- names.tv
		t <- 1
		cat("Counting occurrences in the tv show\n")
		while(t==1 || gs.tv[[t-1]]$file!=file.tv)
		{	#cat("  Processing network \"",gs.tv[[t]]$file,"\"\n",sep="")
			chars <- V(gs.tv[[t]])$name
			occ.tv[chars] <- occ.tv[chars] + 1
			t <- t + 1
		}
		print(occ.tv)
		
		# compute importance based on occurrences
		all.char.names <- sort(unique(c(
			unlist(sapply(gs.nv, function(g) V(g)$name)),
			unlist(sapply(gs.cx, function(g) V(g)$name)),
			unlist(sapply(gs.tv, function(g) V(g)$name))
		)))
		imp.mat <- matrix(NA, nrow=length(all.char.names), ncol=3)
		rownames(imp.mat) <- all.char.names
		colnames(imp.mat) <- c("Novels","Comics","TV Show")
		# we use occurrences as a proxy for importance
		imp.mat[match(names(occ.nv), all.char.names), "Novels"] <- occ.nv/max(occ.nv)
		imp.mat[match(names(occ.cx), all.char.names), "Comics"] <- occ.cx/max(occ.cx)
		imp.mat[match(names(occ.tv), all.char.names), "TV Show"] <- occ.tv/max(occ.tv)
		imp.moy <- apply(imp.mat,1,function(v) mean(v,na.rm=TRUE))
		majority <- apply(imp.mat,1,function(row) length(which(!is.na(row)))>1)
		ranked.chars <- all.char.names[order(majority, imp.moy,decreasing=TRUE)]
				
		# export for later use
		char.importance <- data.frame(all.char.names, imp.mat, imp.moy)
		char.importance <- char.importance[order(majority, imp.moy,decreasing=TRUE),]
		rownames(char.importance) <- NULL
		colnames(char.importance) <- c("Name","Novels","Comics","TVshow","Mean")
		tab.file <- file.path("in",paste0("ranked_importance_S",s,".csv"))
		cat("Exporting as \"",tab.file,"\"\n",sep="")
		write.csv(x=char.importance, file=tab.file, row.names=FALSE, fileEncoding="UTF-8")
	}
}
