# Computes and plots the topological measures of the static graphs.
# 
# Vincent Labatut
# 03/2024
#
# setwd("~/eclipse/workspaces/Networks/NaNet")
# setwd("C:/Users/Vincent/Eclipse/workspaces/Networks/Sachan")
# source("src/descript/static_props.R")
###############################################################################
library("igraph")
library("scales")

source("src/common/topo_measures.R")




###############################################################################
# parameters
TOP_CHAR_NBR <- 20				# number of important characters
NARRATIVE_PART <- 2				# take the whole narrative (0) or only the first two (2) or five (5) narrative units

# measures
measures <- names(TOPO_MEAS_ALL)
#measures <- setdiff(measures, c("communities","w_communities"))

CHARSETS <- c("all","named","common","top")
cs.names <- c("all"="All Characters","named"="Named Characters", "common"="Common Characters", "top"=paste0(TOP_CHAR_NBR," Most Important Characters"))




###############################################################################
# load the static graphs and rank the characters by importance
source("src/common/load_static_nets.R")




###############################################################################
# identify common characters
for(i in 1:length(gs))
{	nm <- V(gs[[i]])$name
	if(i==1)
		common.names <- nm
	else
		common.names <- intersect(common.names, nm)
}




###############################################################################
# also load the graph including all characters
{	if(NARRATIVE_PART==0)
	{	file.nv <- "in/novels/cumul/5.ADwD_72_cumul.graphml"
		file.cx <- "in/comics/cumul/scene/cum_1437.graphml"
		file.tv <- "in/tvshow/cumul/scene/cumulative_4164.graphml"
	}
	else if(NARRATIVE_PART==2)
	{	file.nv <- "in/novels/cumul/2.ACoK_69_cumul.graphml"
		file.cx <- "in/comics/cumul/scene/cum_1437.graphml"
		file.tv <- "in/tvshow/cumul/scene/cumulative_0753.graphml"
	}
	else if(NARRATIVE_PART==5)
	{	file.nv <- "in/novels/cumul/5.ADwD_72_cumul.graphml"
		file.cx <- NA
		file.tv <- "in/tvshow/cumul/scene/cumulative_2248.graphml"
	}
}

# read the chapter-based novel static graph
g.all.nv <- read.graph(file.nv, format="graphml")
E(g.all.nv)$weight <- E(g.all.nv)$weight/max(E(g.all.nv)$weight)
# read the scene-based comics static graph
if(NARRATIVE_PART<5)
{	g.all.cx <- read.graph(file.cx, format="graphml")
	E(g.all.cx)$weight <- E(g.all.cx)$weight/max(E(g.all.cx)$weight)
}
# read the episode-based tvshow static graph
g.all.tv <- read.graph(file.tv, format="graphml")
E(g.all.tv)$weight <- E(g.all.tv)$weight/max(E(g.all.tv)$weight)

# build the list of graphs
{	if(NARRATIVE_PART<5)
		gs.all <- list("novels"=g.all.nv, "comics"=g.all.cx, "tvshow"=g.all.tv)
	else
		gs.all <- list("novels"=g.all.nv, "tvshow"=g.all.tv)
}




###############################################################################
# compute the measures

# loop over character sets
for(charset in CHARSETS)
{	cat("Computing character set ",charset,"\n",sep="")
	
	# output folder
	{	if(charset=="top")
			comm.folder <- paste0(charset,TOP_CHAR_NBR)
		else
			comm.folder <- charset
		narr.folder <- paste0("U",NARRATIVE_PART)
	}
	out.folder <- file.path("out", "descript", "static")
	dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)
	base.file <- file.path(out.folder, paste0(comm.folder, "_", narr.folder))
	
	# result table
	tab <- matrix(NA, nrow=length(gs), ncol=length(measures))
	rownames(tab) <- g.names
	colnames(tab) <- measures
	
	# loop over narratives
	for(i in 1:length(gs))
	{	cat("..Computing narrative ",g.names[i]," (",charset,")\n",sep="")
		
		# possibly use all characters
		if(charset=="all")
			g <- gs.all[[i]]
		else
		{	g <- gs[[i]]
			
			# possibly leep only common characters
			if(charset=="common")
			{	nm <- V(g)$name
				g <- delete_vertices(graph=g, v=which(!(nm %in% common.names)))
			}
			# or possibly keep only top characters
			else if(charset=="top")
			{	nm <- V(g)$name
				g <- delete_vertices(graph=g, v=which(!(nm %in% ranked.chars[1:TOP_CHAR_NBR])))
			}
		}
		
		# loop over measures
		for(meas in measures)
		{	cat("....Computing measure ",meas," (",charset,"-",g.names[i],")\n",sep="")
			
			# compute measure
			mm <- TOPO_MEAS_ALL[[meas]]
			val <- mm$foo(g)
			
			# probably an empty graph
			if(length(val)==0 || all(is.na(val) | is.nan(val) | is.infinite(val)) && gsize(g)<2)
				val <- NA
			# regular case
			else
			{	if(mm$type %in% c("vertex","edge"))
				{	val[is.nan(val) | is.infinite(val)] <- NA
					val <- mean(val,na.rm=TRUE)
				}
			}
			
			# update table
			tab[i,meas] <- val
		}
	}
	
	# record table
	tab.file <- paste0(base.file, "_", charset, ".csv")
	write.csv(x=tab, file=tab.file, row.names=TRUE, fileEncoding="UTF-8")
	cat("..Stats recorded in file \"",tab.file,"\"\n",sep="")
}
