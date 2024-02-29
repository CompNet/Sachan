# Loads the cumulative dynamic graphs for the first two books of novels and comics.
# 
# Author: Vincent Labatut
# 08/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/common/load_dynamic_nets.R")
###############################################################################




###############################################################################
# parameters
NU_NV <- "chapter"	# no choice here

{	# cumultaive vs. instant
	if(CUMULATIVE)
	{	folder <- "cumul"
		pref.nv <- "cumul"
		pref.cx <- "cum"
		pref.tv <- "cumulative"
	}
	else
	{	folder <- "instant"
		pref.nv <- "instant"
		pref.cx <- "inst"
		pref.tv <- "instant"
	}
}

# notes:
# - comics:
#   - last volume 2 chapter: #143 (file  xxxx_143.graphml)
#   - last volume 2 scene:  #1438 (file xxxx_1437.graphml)
# - tv show:
#   - last S02 episode:  #20 (file xxxx_019.graphml)
#   - last S02 block:   #236 (file xxxx_235.graphml)
#   - last S02 scene:   #754 (file xxxx_0753.graphml)
#   - last S05 episode:  #50 (file xxxx_049.graphml)
#   - last S05 block:   #503 (file xxxx_502.graphml)
#   - last S05 scene:  #2249 (file xxxx_2248.graphml)
#   - very last episode: #73 (file xxxx_072.graphml)
#   - very last block:  #739 (file xxxx_738.graphml)
#   - very last scene: #4165 (file xxxx_4164.graphml)
# - novels:
#   - last book 2 chapter: 2.ACoK_69_xxxx.graphml
#   - last book 5 chapter: 5.ADwD_72_xxxx.graphml

{	# temporal coverage
	if(NARRATIVE_PART==0)
	{	# novels
		file.nv <- file.path("in/novels",folder,paste0("5.ADwD_72_",pref.nv,".graphml"))
		# comics
		if(NU_CX=="chapter")
			file.cx <- file.path("in/comics",folder,paste0("chapter/",pref.cx,"_143.graphml"))
		else
			file.cx <- file.path("in/comics",folder,paste0("scene/",pref.cx,"_1437.graphml"))
		# tv show
		if(NU_TV=="episode")
			file.tv <- file.path("in/tvshow",folder,paste0("episode/",pref.tv,"_72.graphml"))
		else if(NU_TV=="block")
			file.tv <- file.path("in/tvshow",folder,paste0("block_locations/",pref.tv,"_738.graphml"))
		else if(NU_TV=="scene")
			file.tv <- file.path("in/tvshow",folder,paste0("scene/",pref.tv,"_4164.graphml"))
	}
	else if(NARRATIVE_PART==2)
	{	# novels
		file.nv <- file.path("in/novels",folder,paste0("2.ACoK_69_",pref.nv,".graphml"))
		# comics
		if(NU_CX=="chapter")
			file.cx <- file.path("in/comics",folder,paste0("chapter/",pref.cx,"_143.graphml"))
		else
			file.cx <- file.path("in/comics",folder,paste0("scene/",pref.cx,"_1437.graphml"))
		# tv show
		if(NU_TV=="episode")
			file.tv <- file.path("in/tvshow",folder,paste0("episode/",pref.tv,"_019.graphml"))
		else if(NU_TV=="block")
			file.tv <- file.path("in/tvshow",folder,paste0("block_locations/",pref.tv,"_235.graphml"))
		else if(NU_TV=="scene")
			file.tv <- file.path("in/tvshow",folder,paste0("scene/",pref.tv,"_0753.graphml"))
	}
	else if(NARRATIVE_PART==5)
	{	# novels
		file.nv <- file.path("in/novels",folder,paste0("5.ADwD_72_",pref.nv,".graphml"))
		# comics
		file.cx <- NA
		# tv show
		if(NU_TV=="episode")
			file.tv <- file.path("in/tvshow",folder,paste0("episode/",pref.tv,"_049.graphml"))
		else if(NU_TV=="block")
			file.tv <- file.path("in/tvshow",folder,paste0("block_locations/",pref.tv,"_502.graphml"))
		else if(NU_TV=="scene")
			file.tv <- file.path("in/tvshow",folder,paste0("scene/",pref.tv,"_2248.graphml"))
	}
}

# test
#NU_NV <- "chapter"
#NU_CX <- "scene"
#NU_TV <- "scene"





###############################################################################
# retrieve the characters' affiliations
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
# clean up a bit
aff.map <- char.tab[,"AllegianceBoth"]
names(aff.map) <- char.tab[,"Name"]
aff.map[aff.map==""] <- "Unknown"
aff.map <- sapply(strsplit(x=aff.map, split=",", fixed=TRUE), function(v) v[1])




###############################################################################
# load the dynamic graphs
gs.all <- list()

# read the novel graphs
gs.nv <- list()
names.nv <- c()
files <- sort(list.files(path=file.path("in/novels",folder), pattern=".+\\.graphml", full.names=TRUE))
i <- 1
last <- FALSE
while(!last)
{	cat("..Loading file \"",files[i],"\"\n",sep="")
	last <- files[i]==file.nv
	
	# load graph
	g.nv <- read.graph(files[i], format="graphml")
	g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)			# keep only named characters
	E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)			# normalize weights
	
	# add book and chapter numbers
	cst <- nchar(file.path("in/novels",folder)) + 1
	book <- as.integer(substr(files[i], start=1+cst, stop=1+cst))
	g.nv <- set_graph_attr(graph=g.nv, name="book", value=book)
	chapter <- as.integer(substr(files[i], start=8+cst, stop=9+cst)) + 1	# +1 bc of the prologues
	g.nv <- set_graph_attr(graph=g.nv, name="chapter", value=chapter)
	
	# add affiliation
	aff <- aff.map[V(g.nv)$name]
	aff[is.na(aff)] <- "Unknown"
	V(g.nv)$affiliation <- aff
	
	# retrieve character names
	names.nv <- sort(union(names.nv, V(g.nv)$name))
	
	gs.nv <- c(gs.nv, list(g.nv))
	i <- i + 1
}
gs.all$novels <- gs.nv
cat("Loaded a total of ",length(gs.nv)," novel networks\n",sep="")

# possibly read the comics graphs
if(!is.na(file.cx))
{	chap <- 1
	vol <- 1
	gs.cx <- list()
	names.cx <- c()
	files <- sort(list.files(path=file.path("in/comics",folder,NU_CX), pattern=".+\\.graphml", full.names=TRUE))
	i <- 1
	last <- FALSE
	while(!last)
	{	cat("..Loading file \"",files[i],"\"\n",sep="")
		last <- files[i]==file.cx
		
		# load graph
		g.cx <- read.graph(files[i], format="graphml")
		g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)		# keep only named characters
		if(gsize(g.cx)>0)
			E(g.cx)$weight <- E(g.cx)$weight/max(E(g.cx)$weight)	# normalize weights
		
		# add volume and chapter numbers
		if(i==74 && NU_CX=="chapter" || i==682 && NU_CX=="scene")
		{	vol <- 2
			chap <- 1
		}
		g.cx <- set_graph_attr(graph=g.cx, name="volume", value=vol)
		g.cx <- set_graph_attr(graph=g.cx, name="chapter", value=chap)
		chap <- chap + 1
		
		# add affiliation
		aff <- aff.map[V(g.cx)$name]
		aff[is.na(aff)] <- "Unknown"
		V(g.cx)$affiliation <- aff
		
		# retrieve character names
		names.cx <- sort(union(names.cx, V(g.cx)$name))
		
		gs.cx <- c(gs.cx, list(g.cx))
		i <- i + 1
	}
	gs.all$comics <- gs.cx
	cat("Loaded a total of ",length(gs.cx)," comic networks\n",sep="")
}

## read the TV Show graphs 
gs.tv <- list()
names.tv <- c()
files <- sort(list.files(path=file.path("in/tvshow",folder,NU_TV), pattern=".+\\.graphml", full.names=TRUE))
i <- 1
last <- FALSE
while(!last)
{	cat("..Loading file \"",files[i],"\"\n",sep="")
	last <- files[i]==file.tv
	
	# load graph
	g.tv <- read.graph(files[i], format="graphml")
	if(gorder(g.tv)>0)
		g.tv <- delete_vertices(graph=g.tv, v=!V(g.tv)$named)			# keep only named characters
	if(gsize(g.tv)>0)
		E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)			# normalize weights
	
	# add affiliation
	aff <- aff.map[V(g.tv)$name]
	aff[is.na(aff)] <- "Unknown"
	V(g.tv)$affiliation <- aff
	
	# retrieve character names
	names.tv <- sort(union(names.tv, V(g.tv)$name))
	
	gs.tv <- c(gs.tv, list(g.tv))
	i <- i + 1
}
gs.all$tvshow <- gs.tv
cat("Loaded a total of ",length(gs.tv)," TV show networks\n",sep="")




###############################################################################
# names of the narratives (for plots)
narr.names <- c("comics"="Comics", "novels"="Novels", "tvshow"="TV Show")
g.names <- names(gs.all)




###############################################################################
# load the chapter mapping file
tab <- read.csv(file="in/comics/chapters.csv", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
chap.map <- tab[,"Rank"]
# re-order comic networks to match the chapter order of the novels 
if(!CUMULATIVE)	# cannot apply to cumulative nets, as they are not built in the proper order
{	gs.cx2 <- list()
	for(n in 1:length(chap.map))
		gs.cx2[[chap.map[n]]] <- gs.cx[[n]]
	gs.cx <- gs.cx2
	gs.all$comics <- gs.cx2
}




###############################################################################
# set the normalized narrative units, based on books/volumes/seasons

# novels
books <- sapply(gs.nv, function(g) graph_attr(g,"book"))
book.nbr <- max(books)
t <- 1
for(b in 1:book.nbr)
{	unit.nbr <- length(which(books==b))
	for(u in 1:unit.nbr)
	{	timestamp <- b + (u-1)/unit.nbr
		gs.nv[[t]] <- set_graph_attr(graph=gs.nv[[t]], name="timestamp", value=timestamp)
		t <- t + 1
	}
}
print(sapply(gs.nv, function(g) graph_attr(graph=g, name="timestamp")))

# comics
volumes <- sapply(gs.cx, function(g) graph_attr(g,"volume"))
volume.nbr <- max(volumes)
t <- 1
for(v in 1:volume.nbr)
{	unit.nbr <- length(which(volumes==v))
	for(u in 1:unit.nbr)
	{	timestamp <- v + (u-1)/unit.nbr
		gs.cx[[t]] <- set_graph_attr(graph=gs.cx[[t]], name="timestamp", value=timestamp)
		t <- t + 1
	}
}
print(sapply(gs.cx, function(g) graph_attr(graph=g, name="timestamp")))

# tv show
seasons <- sapply(gs.tv, function(g) graph_attr(g,"season"))
season.nbr <- max(seasons)
t <- 1
for(s in 1:season.nbr)
{	unit.nbr <- length(which(seasons==s))
	for(u in 1:unit.nbr)
	{	timestamp <- s + (u-1)/unit.nbr
		gs.tv[[t]] <- set_graph_attr(graph=gs.tv[[t]], name="timestamp", value=timestamp)
		t <- t + 1
	}
}
print(sapply(gs.tv, function(g) graph_attr(graph=g, name="timestamp")))
