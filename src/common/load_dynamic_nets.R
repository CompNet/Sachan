# Loads the cumulative dynamic graphs.
# 
# Author: Vincent Labatut
# 08/2023
###############################################################################




###############################################################################
# parameters
{	if(CUMULATIVE)
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

# read the chapter-based novel cumulative graphs
gs.nv <- list()
files <- sort(list.files(path=file.path("in/novels",folder), pattern=".+\\.graphml", full.names=TRUE))
i <- 1
while(files[i]!=file.path("in/novels",folder,paste0("3.ASoS_00_",pref.nv,".graphml")))
{	cat("..Loading file \"",files[i],"\"\n",sep="")
	g.nv <- read.graph(files[i], format="graphml")
	g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)			# keep only named characters
	E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)			# normalize weights
	gs.nv <- c(gs.nv, list(g.nv))
	i <- i + 1
}
cat("Loaded a total of ",length(gs.nv)," novel networks\n",sep="")

# read the chapter-based comics cumulative graphs
gs.cx <- list()
files <- sort(list.files(path=file.path("in/comics",folder,"chapter"), pattern=".+\\.graphml", full.names=TRUE))
for(i in 1:length(files))
{	cat("..Loading file \"",files[i],"\"\n",sep="")
	g.cx <- read.graph(files[i], format="graphml")
	g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)			# keep only named characters
	E(g.cx)$weight <- E(g.cx)$Occurrences/max(E(g.cx)$Occurrences)	# normalize weights
	gs.cx <- c(gs.cx, list(g.cx))
}
cat("Loaded a total of ",length(gs.cx)," comic networks\n",sep="")

# read the episode-based tvshow cumulative graphs # TODO pb: not the same number of time slice as novels and comics
#gs.tv <- list
#files <- sort(list.files(path=file.path("in/tvshow",folder,"episode"), pattern=".+\\.graphml", full.names=TRUE))
#i <- 1
#while(files[i]!=file.path("in/tvshow/",folder,"episode",paste0(pref.tv,"_019.graphml")))
#{	cat("..Loading file \"",files[i],"\"\n",sep="")
#	g.tv <- read.graph(files[i], format="graphml")
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
# re-order comic networks to match the chapter order of the novels 
if(!CUMULATIVE)	# cannot apply to cumulative nets, as they are not built in the proper order
{	gs.cx2 <- list()
	for(n in 1:length(chap.map))
		gs.cx2[[chap.map[n]]] <- gs.cx[[n]]
	gs.cx <- gs.cx2
}
