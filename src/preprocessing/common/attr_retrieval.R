# Retrieves attributes for all characters on the Wiki site:
# https://awoiaf.westeros.org/index.php/List_of_characters
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/common/attr_retrieval.R")
###############################################################################
library("XML")




###############################################################################
base.url <- "https://awoiaf.westeros.org/index.php/"
#char.name <- "A certain man"

# get the list of characters
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)

# get the list of characters to ignore
stop.file <- "in/char_stop.csv"
ignore <- read.csv(stop.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)

# fields to consider
fields <- c("Allegiance"="Allegiance", "Allegiances"="Allegiance", "Race"="Race", "Races"="Race", "Culture"="Culture", "Cultures"="Culture")

# process each character
for(i in 1:nrow(char.tab))
{	char.name <- char.tab[i,"Name"]
	
	if(char.name %in% ignore[,"Name"])
		cat("Ignoring character \"",char.name,"\" (",i,"/",nrow(char.tab),")\n",sep="")
	else
	{	cat("Processing character \"",char.name,"\" (",i,"/",nrow(char.tab),")\n",sep="")
	
		# retrieve the character's wiki page
		url.name <- gsub(" ", "_",char.name)
#		url.name <- gsub("'", "%27",url.name)
		url.name <- URLencode(url.name,reserved=TRUE)
		url <- paste0(base.url,url.name)	# https://awoiaf.westeros.org/index.php/A_certain_man
		html <- tryCatch(expr={readLines(con=url)}, 
				error=function(e) NA)
		if(all(is.na(html)))
			cat("..ERROR: could not load",url,"\n")
		else
		{	dom <- htmlParse(html, encoding="UTF-8")
		
			# retrieving all fields
			for(field in names(fields))
			{	xpath <- paste0("//table[@class='infobox']//tr/th[text()='",field,"']/following-sibling::td//a/text()")
				text <- xmlValue(dom[xpath])
				if(length(text)>0 && !all(is.na(text)))
				{	text <- gsub("\\[.+\\]", "",text)
					text <- text[nchar(text)>0]
					text <- paste(text,collapse=", ")
					cat("..Found",field,":",text,"\n")
					char.tab[i,fields[field]] <- text
				}
			}
		}
	}
}

# write char table
write.csv(x=char.tab, file=char.file, row.names=FALSE, fileEncoding="UTF-8")

# display stats for manual adjustments
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
table(unlist(strsplit(char.tab[,"Allegiance"],", ")))
table(unlist(strsplit(char.tab[,"Culture"],", ")))
table(unlist(strsplit(char.tab[,"Race"],", ")))




###############################################################################
# the GoT extra characters were added manually, 
# and now we do the same thing with the GoT wiki
base.url <- "https://gameofthrones.fandom.com/wiki/"

# get the list of characters
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
colnames(char.tab)[which(colnames(char.tab)=="Allegiance")] <- "AllegianceASOIAF"
colnames(char.tab)[which(colnames(char.tab)=="Culture")] <- "CultureASOIAF"

# add supplementary columns
#addmat <- matrix(NA,nrow=nrow(char.tab), ncol=2)
#colnames(addmat) <- c("AllegianceGoT","CultureGoT")
#char.tab <- cbind(char.tab, addmat)

# get conversion map for tv series
map.file <- "in/tvshow/charmap.csv"
map <- read.csv(map.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
idx <- which(map[,"NormalizedName"]=="")
map[idx,"NormalizedName"] <- map[idx,"TvShowName"]

# fields to consider
fields <- c("Allegiance"="Allegiance", "Allegiances"="Allegiance", "Culture"="Culture", "Cultures"="Culture")

# process each character name
for(i in 1:nrow(char.tab))
{	norm.name <- char.tab[i,"Name"]
	char.name <- map[which(map[,"NormalizedName"]==norm.name),"TvShowName"]
	
	if(length(char.name)==0)
		cat("No match for character \"",norm.name,"\" (",i,"/",nrow(char.tab),")\n",sep="")
	else
	{	cat("Processing character \"",norm.name,"\"/\"",char.name,"\" (",i,"/",nrow(char.tab),")\n",sep="")
		
		# retrieve the character's wiki page
		url.name <- gsub(" ", "_",char.name)
#		url.name <- gsub("'", "%27",url.name)
		url.name <- URLencode(url.name,reserved=TRUE)
		url <- paste0(base.url,url.name)	# https://gameofthrones.fandom.com/wiki/Aegon_I_Targaryen
		html <- tryCatch(expr={readLines(con=url)}, 
				error=function(e) NA)
		if(all(is.na(html)))
			cat("..ERROR: could not load",url,"\n")
		else
		{	dom <- htmlParse(html, encoding="UTF-8")
			
			# retrieving all fields
			for(field in names(fields))
			{	xpath <- paste0("//aside[contains(@class,'portable-infobox')]//div/h3[text()='",field,"']/following-sibling::div//a/text()")
				text <- xmlValue(dom[xpath])
				if(length(text)>0 && !all(is.na(text)))
				{	text <- gsub("\\[.+\\]", "",text)
					text <- text[nchar(text)>0]
					text <- paste(text,collapse=", ")
					cat("..Found",field,":",text,"\n")
					char.tab[i,fields[field]] <- text
				}
			}
		}
	}
}

# write char table
write.csv(x=char.tab, file=char.file, row.names=FALSE, fileEncoding="UTF-8")

# display stats for manual adjustments
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
table(unlist(strsplit(char.tab[,"AllegianceGoT"],", ")))
table(unlist(strsplit(char.tab[,"CultureGoT"],", ")))




###############################################################################
# additional processing
table(c(unlist(strsplit(char.tab[,"AllegianceASOIAF"],", ")),unlist(strsplit(char.tab[,"AllegianceGoT"],", "))))
table(c(unlist(strsplit(char.tab[,"CultureASOIAF"],", ")),unlist(strsplit(char.tab[,"CultureGoT"],", "))))
