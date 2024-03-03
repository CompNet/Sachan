# Computes and plots the evolution of numbers of gender-based triangles.
# It is based on evol_props.R.
# 
# Vincent Labatut
# 03/2024
#
# setwd("~/eclipse/workspaces/Networks/NaNet")
# setwd("C:/Users/Vincent/Eclipse/workspaces/Networks/Sachan")
# source("src/gender/evol_gender.R")
###############################################################################
library("igraph")
library("scales")

source("src/common/colors.R")
source("src/gender/gender_measures.R")




###############################################################################
# parameters
CUMULATIVE <- TRUE				# use the instant (FALSE) or cumulative (TRUE) networks
WINDOW_SIZE <- 3				# for the instant mode (cf. above), size of the window used for smoothing 
TOP_CHAR_NBR <- 20				# number of important characters
NU_NV <- "chapter"				# novel narrative unit: no choice here
NU_CX <- "chapter"				# comics narrative unit: chapter, scene
NU_TV <- "episode"				# tv narrative unit: episode, block, scene

CHARSETS <- c("named","common","top")
cs.names <- c("named"="Named Characters", "common"="Common Characters", "top"=paste0(TOP_CHAR_NBR," Most Important Characters"))




###############################################################################
# always take the full narratives
NARRATIVE_PART <- 0
# load the static graphs and rank the characters by importance
source("src/common/load_static_nets.R")
# load the dynamic graphs
source("src/common/load_dynamic_nets.R")
# start/end narrative units
start.nu <- 1
{	if(NARRATIVE_PART==0)
		end.nu <- 8
	else 
		end.nu <- NARRATIVE_PART
}




###############################################################################
# identify common characters in the static networks
for(i in 1:length(gs))
{	nm <- V(gs[[i]])$name
	if(i==1)
		common.names <- nm
	else
		common.names <- intersect(common.names, nm)
}




###############################################################################
# base folder
base.name <- paste0("novels-", NU_NV)
if("comics" %in% names(gs.all))
	base.name <- paste0(base.name, "_comics-", NU_CX)
base.name <- paste0(base.name, "_tvshow-", NU_TV)
# dyn mode folder
{	if(CUMULATIVE)
		dyn.folder <- "cumul"
	else
	{	dyn.folder <- "instant"
		if(WINDOW_SIZE>0)
			dyn.folder <- paste0(dyn.folder,"_smoothed-",WINDOW_SIZE)		
	}
}
# output folder
out.folder <- file.path("out", "gender", dyn.folder, base.name)
dir.create(path=out.folder, showWarnings=FALSE, recursive=TRUE)




###############################################################################
# compute the measures

# TODO scene stats must be implemented here, as they use the whole dyn series

# loop over character sets
list.res <- list()
for(charset in CHARSETS)
{	cat("Computing charecter set ",charset,"\n",sep="")
	
	# loop over narratives
	list.narr <- list()
	for(i in 1:length(gs.all))
	{	cat("..Computing narrative ",g.names[i]," (",charset,")\n",sep="")
		gs <- gs.all[[i]]
		
		# loop over time slices
		list.meas <- list()
		for(t in 1:length(gs))
		{	cat("....Computing time ",t,"/",length(gs),"(",charset,"--",g.names[i],")\n",sep="")
			g <- gs[[t]]
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
			
			# process stats
			stats <- compute.gender.measures(g)
			
			# update vectors
			for(m in 1:length(stats))
			{	mname <- names(stats)[m]
				mat <- stats[[m]]
				for(r in 1:nrow(mat))
				{	rname <- rownames(mat)[r]
					for(c in 1:ncol(mat))
					{	cname <- colnames(mat)[c]
						val <- mat[r,c]
						meas <- paste0(mname,":",rname,":",cname)
						if(t==1)
							list.meas[[meas]] <- val
						else
							list.meas[[meas]] <- c(list.meas[[meas]], val)
					}
				}
			}
		}
		
		# store results
		list.narr[[g.names[i]]] <- list.meas
	}
	
	# store results
	list.res[[charset]] <- list.narr
}




###############################################################################
# plot each narrative as a distinct series
colors <- brewer_pal(type="qual", palette=2)(length(gs.all))
measures <- c(
	"fmratio"="edges:All:F-/M- Ratio",
	"homophily"="edges:All:Homophily",
	"gdratio"="degree:All:GenderDegreeRatio",
	"gsratio"="strength:All:GenderStrengthRatio"
)
meas.fullnames <- c(
	"fmratio"="F/M Edge Ratio",
	"homophily"="Gender Homophily",
	"gdratio"="Gender Degree Ratio",
	"gsratio"="Gender Strength Ratio"
) 

# loop over character sets
for(charset in CHARSETS)
{	cat("Plotting character set ",charset,"\n",sep="")
	xlab <- "Time (Novel/Volume/Season)"
	
	# loop over measures
	for(m in 1:length(measures))
	{	cat("..Plotting measure ",measures[m]," (",charset,")\n",sep="")
		
		# file/label name
		meas <- names(measures)[m]
		ylab <- meas.fullnames[meas]
		
		# create folder
		local.folder <- file.path(out.folder, meas)
		dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
		
		# clean values and compute range
		rg <- c()
		list.vals <- list()
		for(i in 1:length(gs.all))
		{	vals <- list.res[[charset]][[g.names[i]]][[measures[m]]]
			vals[is.nan(vals) | is.infinite(vals)] <- NA
			rg <- range(rg,vals,na.rm=TRUE)
			list.vals[[i]] <- vals
		}
		
		plot.file <- file.path(local.folder, paste0("narrative-all_charset-",charset))
		pdf(paste0(plot.file,".pdf"), width=12, height=7)	# bg="white"
			par(mar=c(5, 4, 4-2.50, 2-1.25)+0.1)	# margins Bottom Left Top Right
			plot(
				NULL, 
				main=cs.names[charset], xlab=xlab, ylab=ylab,
				xlim=c(start.nu,end.nu+1), ylim=rg
			)
			# vertical lines
			abline(v=3, lty=3)
			abline(v=6, lty=3)
			# loop over narratives
			for(i in 1:length(gs.all))
			{	xs <- sapply(gs.all[[i]], function(g) graph_attr(g,"timestamp"))
				ys <- list.vals[[i]]
				if(!CUMULATIVE && WINDOW_SIZE>0)
					ys <- sapply(1:length(ys), function(j) mean(ys[max(1,round(j-WINDOW_SIZE/2)):min(length(ys),round(j+WINDOW_SIZE/2))]))
				lines(x=xs, y=ys, col=colors[i], lwd=2)
			}
			# add legend
			legend(
				x="bottomright",
				legend=narr.names[g.names],
				fill=colors
			)
		dev.off()
	}	
}




###############################################################################
# plot each character set as a distinct series
unit.map <- c("novels"="Books", "comics"="Volumes", "tvshow"="Seasons")
cs.legend <- CHARSETS
cs.legend[cs.legend=="top"] <- paste0("top",TOP_CHAR_NBR)
		
# loop over narratives
for(i in 1:length(gs.all))
{	cat("Plotting narrative ",g.names[i],"\n",sep="")
	xlab <- paste0("Time (",unit.map[g.names[i]],")")
	xs <- sapply(gs.all[[i]], function(g) graph_attr(g,"timestamp"))
	cols <- c(combine.colors(col1=colors[i], col2="WHITE", transparency=67), colors[i], combine.colors(col1=colors[i], col2="BLACK", transparency=67)) 
	
	# loop over measures
	for(m in 1:length(measures))
	{	cat("..Plotting measure ",measures[m]," (",g.names[i],")\n",sep="")
		
		# file/label name
		meas <- names(measures)[m]
		ylab <- meas.fullnames[meas]
		
		# create folder
		local.folder <- file.path(out.folder, meas)
		dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
		
		# clean values and compute range
		rg <- c()
		list.vals <- list()
		for(c in 1:length(CHARSETS))
		{	charset <- CHARSETS[c]
			vals <- list.res[[charset]][[g.names[i]]][[measures[m]]]
			vals[is.nan(vals) | is.infinite(vals)] <- NA
			rg <- range(rg,vals,na.rm=TRUE)
			list.vals[[c]] <- vals
		}
		
		colors <- brewer_pal(type="qual", palette=2)(length(gs.all))
		plot.file <- file.path(local.folder, paste0("narrative-",g.names[i],"_charset-all"))
		pdf(paste0(plot.file,".pdf"), width=12, height=7)	# bg="white"
			par(mar=c(5, 4, 4-2.50, 2-1.25)+0.1)	# margins Bottom Left Top Right
			plot(
				NULL, 
				main=bquote(bolditalic(.(narr.names[g.names[i]]))), 
				xlab=xlab, ylab=ylab,
				xlim=c(start.nu,round(max(xs))), ylim=rg
			)
			# vertical lines
			abline(v=3, lty=3)
			abline(v=6, lty=3)
			# loop over character sets
			for(c in 1:length(CHARSETS))
			{	ys <- list.vals[[c]]
				if(!CUMULATIVE && WINDOW_SIZE>0)
					ys <- sapply(1:length(ys), function(j) mean(ys[max(1,round(j-WINDOW_SIZE/2)):min(length(ys),round(j+WINDOW_SIZE/2))]))
				lines(x=xs, y=ys, col=cols[c], lty=c, lwd=2)
			}
			# add legend
			legend(
				x="bottomright",
				legend=cs.legend,
				col=cols, lty=1:length(CHARSETS), lwd=2,
				bg="#FFFFFFCC"
			)
		dev.off()
	}	
}
