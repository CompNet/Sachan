# Reports matching performances obtained for certain characters of interest.
# 
# Author: Vincent Labatut
# 02/2024
# 
# setwd("D:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/specific_chars.R")
###############################################################################
library("igraph")




###############################################################################
# processing parameters
CHARSET <- "named"			# all named characters (named), or only those common to both compared graphs (common), or the 20 most important (top)




###############################################################################
merged.chars <- matrix(c(
	"Lyn Corbray","Lucas Corbray",
	"Harrion Karstark","Arnolf Karstark",
	"Benerro","Moqorro",
	"Meribald","Elder Brother"
),ncol=2,byrow=TRUE)
rownames(merged.chars) <- c(
	"Vance Corbray",
	"Alys Karstark",
	"Kinvara",
	"Ray"
)
#
subs.chars <- c(
	"Jeyne Westerling"="Talisa Stark",
	"Vargo Hoat"="Locke",
	"Cleos Frey"="Alton Lannister",
	"Asha Greyjoy"="Yara Greyjoy",
	"Robert Arryn"="Robin Arryn",
	"Jhogo"="Kovarro",
	"Grazdan mo Eraz"="Razdal mo Eraz",
	"Grazdan mo Ullhor"="Greizhen mo Ullhor",
	"Stalwart Shield"="White Rat"
)




###############################################################################
for(NARRATIVE_PART in c(2,5))
{	cat(">>>>>>>>>> Processing narrative part = ",NARRATIVE_PART," <<<<<<<<<<\n",sep="")
	
	# load the static graphs and rank the characters by importance
	source("src/common/load_static_nets.R")
	
	# load the csv file
	tab.file <- file.path("out", "matching", paste0("first_",NARRATIVE_PART), "jaccard", "named", "novels_vs_tvshow", "sim_matrix_all.csv")
	sim.mat <- read.csv(file=tab.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE, row.names=1)
	
	
	# look for substituted chars
	cat("..Dealing with the substituted characters\n")
	for(i in 1:length(subs.chars))
	{	old.char <- names(subs.chars)[i]
		if(old.char %in% rownames(sim.mat))
		{	cat("......First character:",old.char,"\n")
			
			new.char <- subs.chars[i]
			if(new.char %in% colnames(sim.mat))
			{	cat("......Second character:",new.char,"\n")
				
				self.sim <- sim.mat[old.char,new.char]
				cat("........Self-similarity:",self.sim,"\n")
				#
				ba1 <- which.max(sim.mat[old.char, -which(colnames(sim.mat)==new.char)])
				best.alter1 <- sim.mat[old.char,ba1]
				cat("........Alter similarity 1: ",best.alter1," (",colnames(sim.mat)[ba1],")\n",sep="")
				#
				ba2 <- which.max(sim.mat[-which(colnames(sim.mat)==old.char), new.char])
				best.alter2 <- sim.mat[ba2,new.char]
				cat("........Alter similarity 2: ",best.alter2," (",rownames(sim.mat)[ba2],")\n",sep="")
				#
				cat("......>> Self",if(self.sim>best.alter1 && self.sim>best.alter2) "BETTER" else "worse","than both alters\n")
			}
			else
				cat("......Could not find second character \"",new.char,"\"\n",sep="")
		}
		else
			cat("..Could not find first character \"",old.char,"\"\n",sep="")
	}
	cat("\n")
	
	# look for merged chars
	cat("..Dealing with the merged characters\n")
	for(i in 1:nrow(merged.chars))
	{	old.char <- rownames(merged.chars)[i]
		if(old.char %in% rownames(sim.mat))
		{	cat("......First character:",old.char,"\n")
			
			new.chars <- merged.chars[i,]
			for(j in 1:length(new.chars))
			{	new.char <- new.chars[j]
				if(new.char %in% colnames(sim.mat))
				{	cat("......Second character:",new.char,"\n")
					
					self.sim <- sim.mat[old.char,new.char]
					cat("........Self-similarity:",self.sim,"\n")
					#
					ba1 <- which.max(sim.mat[old.char, -which(colnames(sim.mat)==new.char)])
					best.alter1 <- sim.mat[old.char,ba1]
					cat("........Alter similarity 1: ",best.alter1," (",colnames(sim.mat)[ba1],")\n",sep="")
					#
					ba2 <- which.max(sim.mat[-which(colnames(sim.mat)==old.char), new.char])
					best.alter2 <- sim.mat[ba2,new.char]
					cat("........Alter similarity 2: ",best.alter2," (",rownames(sim.mat)[ba2],")\n",sep="")
					#
					cat("......>> Self",if(self.sim>best.alter1 && self.sim>best.alter2) "BETTER" else "worse","than both alters\n")
				}
				else
					cat("......Could not find second character \"",new.char,"\"\n",sep="")
			}
		}
		else
			cat("..Could not find first character \"",old.char,"\"\n",sep="")
	}
	cat("\n\n")
}
