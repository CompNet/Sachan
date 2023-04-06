# Retrieves attributes for all characters on the Wiki site:
# https://awoiaf.westeros.org/index.php/List_of_characters
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/preprocessing/attr_retrieval.R")
###############################################################################
library("XML")




###############################################################################
base.url <- "https://awoiaf.westeros.org/index.php/"
#char.name <- "A certain man"

# get the list of characters
char.file <- "in/characters.csv"
char.tab <- read.csv(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)

# get the list of characters to ignore
stop.file <- "in/char_stop.csv"
ignore <- read.csv(stop.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)

# field to consider
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

# additional adjustments
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
table(unlist(strsplit(char.tab[,"Allegiance"],", ")))
table(unlist(strsplit(char.tab[,"Culture"],", ")))
table(unlist(strsplit(char.tab[,"Race"],", ")))
