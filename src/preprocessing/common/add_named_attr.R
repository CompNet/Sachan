# Adds a new 'named' vertex attribute to the novels and TV show networks
# (the comics already have this attribute). It shows whether a character
# has a proper noun or nickname (by comparison to a purely descriptive name).
#
# The script also corrects a few errors remaining after the name normalization
# process already conducted before.
# 
# Author: Vincent Labatut
# 05/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/common/add_named_attr.R")
###############################################################################
library("igraph")




###############################################################################
# get the list of characters
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)




###############################################################################
# identify the unnamed characters in the tv show
g.tv <- read.graph("in/tvshow/cumul/episode/cumulative_72.graphml", format="graphml")
names.tv <- sort(V(g.tv)$name)
map <- match(names.tv, char.tab[,"Name"])
idx <- which(is.na(map))
unnamed.tv <- names.tv[idx]
print(unnamed.tv)

# named characters not appearing in the main list
exceptions <- c("Lord Blackmont","Lord Portan")
# character names that need to be fixed
corrections <- c("Olenna Tyrell"="Olenna Redwyne")

# update the xml files with the new attribute
for(time in c("cumul","instant"))
{	cat("Handling time mode '",time,"'\n",sep="")
	
	for(scale in c("episode","scene"))
	{	cat("Handling time scale '",scale,"'\n",sep="")
		
		# list all graphml files in the folder
		net.folder <- file.path("in","tvshow",time,scale)
		files <- list.files(path=net.folder, pattern=".+\\.graphml")
		
		# process each one
		for(file in files)
		{	# read graph
			path <- file.path(net.folder, file)
			cat("..Loading graph \"",path,"\"\n",sep="")
			g <- read.graph(file=path, format="graphml")
			
			# if graph has vertices
			if(gorder(g)>0)
			{	# handle remaining name normalization error
				if(any(V(g)$name %in% names(corrections)))
				{	idx <- which(V(g)$name %in% names(corrections))
					V(g)$name[idx] <- corrections[V(g)$name[idx]]
					cat("....Corrected ",length(idx)," remaining normalization errors\n",sep="")
				}
				
				# add attribute
				named <- rep(FALSE, gorder(g))
				named[V(g)$name %in% exceptions] <- TRUE
				map <- match(V(g)$name, char.tab[,"Name"])
				named[!is.na(map)] <- char.tab[map[!is.na(map)],"Named"]
				V(g)$named <- named
				cat("....Found the following ",length(which(!named))," unnamed characters: ",paste(V(g)$name[!named],collapse=", "),"\n",sep="")
			}
			# if graph is empty 
			else
			{	cat("....Empty graph\n")
				# add dummy vertex, add attribute, remove vertex
				g <- add_vertices(g, 1)
				g <- set_vertex_attr(g, name="named", value=TRUE)
				g <- delete_vertices(g, 1)
			}
			
			# record updated graph
			cat("....Recording graph\n")
			write.graph(g, file=path, format="graphml")
		}
	}
}




###############################################################################
# read the static novel graph
g.nv <- read.graph("in/novels/cumul/5.ADwD_72_cumul.graphml", format="graphml")
names.nv <- sort(V(g.nv)$name)
map <- match(names.nv, char.tab[,"Name"])
idx <- which(is.na(map))
unnamed.nv <- names.nv[idx]
print(unnamed.nv)
	
	# named characters not appearing in the main list
	exceptions <- c(
		"Aewin Targaryen","Lord Cerwyn","Lord Farman","Lord Swann","Lord Corbray","Lord Goodbrook",
		"Lord Whent","Lady Rykker","Lord Crakehall","Lord Harroway","Old Lord Dustin","Ser Morghil","Lord Darry",
		"Lord Lothston","Ser Wilbert","Walda Frey","Lord Dondarrion","Lord Mooton","Walda Frey 3","Grazdan 2",
		"Lord Blackwood", "Lord Estermont", "Lord of Godsgrace", "Walda Frey 4", "Grazdan 3", "Grazdan 4",
		#
		"Lord Brandon"
	)
	# character names that need to be fixed
	corrections <- c(
		"Oswyn Longneck the Thrice-Hanged"="Oswyn",
		"Baelor Targaryen"="Baelor Targaryen (son of Daeron II)",
		"Roland Crakehall"="Roland Crakehall (lord)",
		"Ser Donnel of Duskendale"="Donnel of Duskendale",
		"Daeron Targaryen (Son of Maekar 1)"="Daeron Targaryen (son of Maekar I)",
		"Hareth (Grand Maester)"="Hareth (maester)",
		"Leo Longthorn"="Leo Tyrell (Longthorn)",
		"Byron The Beautiful"="Byron the Beautiful",
		"Dale the Dread"="Dale Drumm",
		"Henly (Maester)"="Henly (maester)",
		"Pisswater Prince"="Pisswater prince",
		"Fat Fellow"="Fat fellow",
		"Septon Utt"="Utt",
		#
		"Archmaester Theobald"="Theobald",
		"Benjen the Bitter"="Benjen Stark (Bitter)",
		"Benjen the Sweet"="Benjen Stark (Sweet)",
		"Brandon the Bad"="Brandon Stark (Bad)",
		"Edderion the Bridegroom"="Edderion Stark",
		"Gorghan"="Gorghan of Old Ghis",
		"Pate"="Pate (novice)",
		"Walton the Moon King"="Walton Stark",
		"Annara Frey"="Annara Farring"
	)

# update the xml files with the new attribute
for(time in c("cumul","instant"))
{	cat("Handling time mode '",time,"'\n",sep="")
	
	# list all graphml files in the folder
	net.folder <- file.path("in","novels",time)
	files <- list.files(path=net.folder, pattern=".+\\.graphml")
	
	# process each one
	for(file in files)
	{	# read graph
		path <- file.path(net.folder, file)
		cat("..Loading graph \"",path,"\"\n",sep="")
		g <- read.graph(file=path, format="graphml")
		
		# remove quotes from strings in every attribute
		for(att in vertex_attr_names(g))
		{	vals <- vertex_attr(g, name=att)
			if(is.character(vals))
				vals <- gsub("\"", "", vals)
			g <- set_vertex_attr(g, name=att, value=vals)
		}
		for(att in edge_attr_names(g))
		{	vals <- edge_attr(g, name=att)
			if(is.character(vals))
				vals <- gsub("\"", "", vals)
			g <- set_edge_attr(g, name=att, value=vals)
		}
		
		# if graph has vertices
		if(gorder(g)>0)
		{	# handle remaining name normalization error
			if(any(V(g)$name %in% names(corrections)))
			{	idx <- which(V(g)$name %in% names(corrections))
				V(g)$name[idx] <- corrections[V(g)$name[idx]]
				cat("....Corrected ",length(idx)," remaining normalization errors\n",sep="")
			}
			
			# remove possible duplicates caused by the name correction
			dups <- unique(V(g)$name[duplicated(V(g)$name)])
			for(dup in dups)
			{	idx <- which(V(g)$name==dup)
				mapping <- 1:gorder(g)
				mapping[idx[-1]] <- idx[1]
				g <- contract(graph=g, mapping=mapping, vertex.attr.comb="first")
				g <-  delete.vertices(graph=g, v=idx[-1])
			}
			# all vertex attributes are now lists, for some reason...
			for(att in vertex_attr_names(g))
			{	vals <- unlist(vertex_attr(g, name=att))
				g <- delete_vertex_attr(g, name=att)
				g <- set_vertex_attr(g, name=att, value=vals)
			}
			
			# check multiple edges
			if(any_multiple(g))
				stop("Must deal with multiple edges (probably using 'simplify')")
			
			# add attribute
			named <- rep(FALSE, gorder(g))
			named[V(g)$name %in% exceptions] <- TRUE
			map <- match(V(g)$name, char.tab[,"Name"])
			named[!is.na(map)] <- char.tab[map[!is.na(map)],"Named"]
			V(g)$named <- named
			cat("....Found the following ",length(which(!named))," unnamed characters: ",paste(V(g)$name[!named],collapse=", "),"\n",sep="")
		}
		# if graph is empty 
		else
		{	cat("....Empty graph\n")
			# add dummy vertex, add attribute, remove vertex
			g <- add_vertices(g, 1)
			g <- set_vertex_attr(g, name="named", value=TRUE)
			g <- delete_vertices(g, 1)
		}

		# record updated graph
		cat("....Recording graph\n")
		write.graph(g, file=path, format="graphml")
	}
}

# post test
g.nv <- read.graph("in/novels/cumul/5.ADwD_72_cumul.graphml", format="graphml")
names.nv <- sort(V(g.nv)$name)
map <- match(names.nv, char.tab[,"Name"])
idx <- which(is.na(map) & !(names.nv %in% exceptions))
unnamed.nv <- names.nv[idx]
print(unnamed.nv)




###############################################################################
# the attribute already exists in the comics nets, 
# but the name is not identical to the others, so it must be changed
for(time in c("cumul","instant"))
{	cat("Handling time mode '",time,"'\n",sep="")
	
	for(scale in c("chapter","scene"))
	{	cat("Handling time scale '",scale,"'\n",sep="")
		
		# list all graphml files in the folder
		net.folder <- file.path("in","comics",time,scale)
		files <- list.files(path=net.folder, pattern=".+\\.graphml")
		
		# process each one
		for(file in files)
		{	# read graph
			path <- file.path(net.folder, file)
			cat("..Loading graph \"",path,"\"\n",sep="")
			g <- read.graph(file=path, format="graphml")
			
			# if graph has vertices
			if(gorder(g)>0)
			{	vals <- V(g)$Named
				g <- delete_vertex_attr(g, name="Named")
				g <- set_vertex_attr(g, name="named", value=vals)
				cat("....Updated attribute name\n")
			}
			# if graph is empty 
			else
			{	cat("....Empty graph\n")
				# add dummy vertex, add attribute, remove vertex
				g <- add_vertices(g, 1)
				g <- set_vertex_attr(g, name="named", value=TRUE)
				g <- delete_vertices(g, 1)
			}
			
			# record updated graph
			cat("....Recording graph\n")
			write.graph(g, file=path, format="graphml")
		}
	}
}
