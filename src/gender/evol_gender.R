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
WINDOW_SIZE <- 0				# for the instant mode (cf. above), size of the window used for smoothing 
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

# TODO Bechdel test missing (requires annotation)

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
		{	cat("....Computing time ",t,"/",length(gs),"(",charset,"-",g.names[i],")\n",sep="")
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
	"vertices-prop-fem"="vertices:Female:FM_Proportion",
	"degree-prop-fem"="degree:Female:FM_Proportion",
	"strength-prop-fem"="strength:Female:FM_Proportion",
	"fmratio"="edges:All:F-/M- Ratio",
	"homophily"="edges:All:Homophily",
	"gdratio"="degree:All:GenderDegreeRatio",
	"gsratio"="strength:All:GenderStrengthRatio"
)
meas.fullnames <- c(
	"vertices-prop-fem"="Proportion of female vertices (%)",
	"degree-prop-fem"="Female proportion of total degree (%)",
	"strength-prop-fem"="Female proportion of total strength (%)",
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
		
		# produce plot
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
				fill=colors,
				bg="#FFFFFFCC"
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
		tab <- matrix(NA,nrow=length(vals),ncol=length(CHARSETS))
		
		# produce plot
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
			{	ys <- tab[,c] <- list.vals[[c]]
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
		
		# record csv
		colnames(tab) <- CHARSETS
		write.csv(x=tab, file=paste0(plot.file,".csv"), row.names=FALSE, fileEncoding="UTF-8")
	}	
}




###############################################################################
# plot groups of related measures
group.meas <- list(
	# vertices
	"vertices-count"=list(
		"fullname"="Number of Vertices",
		"variables"=c("Male"="vertices:Male:Count", "Female"="vertices:Female:Count")),
	"vertices-prop"=list(
		"fullname"="Proportion of Vertices (%)",
		"variables"=c("Male"="vertices:Male:FM_Proportion", "Female"="vertices:Female:FM_Proportion")),
	# edges
	"edges-count"=list(
		"fullname"="Number of Edges",
		"variables"=c("Female-Female"="edges:F-F:Count", "Female-Male"="edges:F-M:Count", "Male-Male"="edges:M-M:Count")),
	"edges-prop"=list(
		"fullname"="Proportion of Edges (%)",
		"variables"=c("Female-Female"="edges:F-F:Proportion", "Female-Male"="edges:F-M:Proportion", "Male-Male"="edges:M-M:Proportion")),
	# degree
	"degree-total"=list(
		"fullname"="Total Degree",
		"variables"=c("Male"="degree:Male:Total","Female"="degree:Female:Total")),
	"degree-prop"=list(
		"fullname"="Proportion of Degree (%)",
		"variables"=c("Male"="degree:Male:FM_Proportion","Female"="degree:Female:FM_Proportion")),
	# strength
	"strength-total"=list(
		"fullname"="Total Strength",
		"variables"=c("Male"="strength:Male:Total","Female"="strength:Female:Total")),
	"strength-prop"=list(
		"fullname"="Proportion of Strength (%)",
		"variables"=c("Male"="strength:Male:FM_Proportion","Female"="strength:Female:FM_Proportion")),
	# density
	"density"=list(
		"fullname"="Density",
		"variables"=c("Male"="density:Male:Density", "Female"="density:Female:Density")),
	# triangles
	"triangles-count"=list(
		"fullname"="Number of Triangles",
		"variables"=c("Female-Female-Female"="triangles:Female-Female-Female:Count", "Female-Female-Male"="triangles:Female-Female-Male:Count", "Female-Male-Male"="triangles:Female-Male-Male:Count", "Male-Male-Male"="triangles:Male-Male-Male:Count")),
	"triangles-prop"=list(
		"fullname"="Proportion of Triangles (%)",
		"variables"=c("Female-Female-Female"="triangles:Female-Female-Female:FM_Proportion", "Female-Female-Male"="triangles:Female-Female-Male:FM_Proportion", "Female-Male-Male"="triangles:Female-Male-Male:FM_Proportion", "Male-Male-Male"="triangles:Male-Male-Male:FM_Proportion"))
)

# loop over narratives
for(i in 1:length(gs.all))
{	cat("Plotting narrative ",g.names[i],"\n",sep="")
	xlab <- paste0("Time (",unit.map[g.names[i]],")")
	xs <- sapply(gs.all[[i]], function(g) graph_attr(g,"timestamp"))
	cols <- c(combine.colors(col1=colors[i], col2="WHITE", transparency=67), colors[i], combine.colors(col1=colors[i], col2="BLACK", transparency=67)) 
	
	# loop over character sets
	for(c in 1:length(CHARSETS))
	{	charset <- CHARSETS[c]
		cat("..Plotting charset ",charset," (",g.names[i],")\n",sep="")
		
		# loop over measure groups
		for(m in 1:length(group.meas))
		{	cat("....Plotting measure group ",names(group.meas)[m]," (",g.names[i],"-",charset,")\n",sep="")
			
			# file/label name
			meas <- names(group.meas)[m]
			ylab <- group.meas[[meas]]$fullname
			variables <- group.meas[[meas]]$variables
			
			# create folder
			local.folder <- file.path(out.folder, meas)
			dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
			
			# clean values and compute range
			rg <- c()
			list.vals <- list()
			for(v in 1:length(variables))
			{	vals <- list.res[[charset]][[g.names[i]]][[variables[v]]]
				vals[is.nan(vals) | is.infinite(vals)] <- NA
				rg <- range(rg,vals,na.rm=TRUE)
				list.vals[[v]] <- vals
			}
			tab <- matrix(NA,nrow=length(vals),ncol=length(variables))
			
			# produce plot
			plot.file <- file.path(local.folder, paste0("narrative-",g.names[i],"_charset-",charset))
			pdf(paste0(plot.file,".pdf"), width=12, height=7)	# bg="white"
				par(mar=c(5, 4, 4-2.50, 2-1.25)+0.1)	# margins Bottom Left Top Right
				plot(
					NULL, 
					main=bquote(bolditalic(.(narr.names[g.names[i]]))~" - "~bold(.(cs.names[charset]))),
					xlab=xlab, ylab=ylab,
					xlim=c(start.nu,round(max(xs))), ylim=rg
				)
				# vertical lines
				abline(v=3, lty=3)
				abline(v=6, lty=3)
				# loop over variables
				for(v in 1:length(variables))
				{	ys <- tab[,v] <- list.vals[[v]]
					if(!CUMULATIVE && WINDOW_SIZE>0)
						ys <- sapply(1:length(ys), function(j) mean(ys[max(1,round(j-WINDOW_SIZE/2)):min(length(ys),round(j+WINDOW_SIZE/2))]))
					lines(x=xs, y=ys, col=ATT_COLORS_SEX[names(variables)[v]], lwd=2)
				}
				# add legend
				legend(
					x="bottomright",
					legend=names(variables),
					col=ATT_COLORS_SEX[names(variables)], lwd=2,
					bg="#FFFFFFCC"
				)
			dev.off()
			
			# record csv
			colnames(tab) <- names(variables)
			write.csv(x=tab, file=paste0(plot.file,".csv"), row.names=FALSE, fileEncoding="UTF-8")
		}
	}	
}




###############################################################################
# plot each narrative as a distinct series, but for pairs of F/M measures
pair.meas <- list(
	# vertices
	"vertices-count"=list(
		"fullname"="Number of Vertices",
		"variables"=c("Male"="vertices:Male:Count", "Female"="vertices:Female:Count")),
	"vertices-prop"=list(
		"fullname"="Proportion of Vertices (%)",
		"variables"=c("Male"="vertices:Male:FM_Proportion", "Female"="vertices:Female:FM_Proportion")),
	# degree
	"degree-total"=list(
		"fullname"="Total Degree",
		"variables"=c("Male"="degree:Male:Total","Female"="degree:Female:Total")),
	"degree-prop"=list(
		"fullname"="Proportion of Degree (%)",
		"variables"=c("Male"="degree:Male:FM_Proportion","Female"="degree:Female:FM_Proportion")),
	# strength
	"strength-total"=list(
		"fullname"="Total Strength",
		"variables"=c("Male"="strength:Male:Total","Female"="strength:Female:Total")),
	"strength-prop"=list(
		"fullname"="Proportion of Strength (%)",
		"variables"=c("Male"="strength:Male:FM_Proportion","Female"="strength:Female:FM_Proportion")),
	# density
	"density"=list(
		"fullname"="Density",
		"variables"=c("Male"="density:Male:Density", "Female"="density:Female:Density"))
)

# loop over character sets
for(charset in CHARSETS)
{	cat("Plotting character set ",charset,"\n",sep="")
	xlab <- "Time (Novel/Volume/Season)"
	
	# loop over measure pairs
	for(m in 1:length(pair.meas))
	{	cat("..Plotting measure pair ",names(pair.meas)[m]," (",charset,")\n",sep="")
		
		# file/label name
		meas <- names(pair.meas)[m]
		ylab <- pair.meas[[meas]]$fullname
		variables <- pair.meas[[meas]]$variables
		
		# create folder
		local.folder <- file.path(out.folder, meas)
		dir.create(path=local.folder, showWarnings=FALSE, recursive=TRUE)
		
		# clean values and compute range
		rg <- c()
		list.vals <- list()
		for(i in 1:length(gs.all))
		{	list.gdr <- list()
			for(v in 1:length(variables))
			{	vals <- list.res[[charset]][[g.names[i]]][[variables[v]]]
				vals[is.nan(vals) | is.infinite(vals)] <- NA
				rg <- range(rg,vals,na.rm=TRUE)
				list.gdr[[v]] <- vals
			}
			list.vals[[i]] <- list.gdr
		}
		
		# produce plot
		plot.file <- file.path(local.folder, paste0("narrative-all_charset-",charset))
		pdf(paste0(plot.file,".pdf"), width=12, height=7)	# bg="white"
			par(mar=c(5, 4, 4-2.50, 2-1.25)+0.1)	# margins Bottom Left Top Right
			plot(
				NULL, 
				main=bquote(bold(.(cs.names[charset]))),
				xlab=xlab, ylab=ylab,
				xlim=c(start.nu,round(max(xs))), ylim=rg
			)
			# vertical lines
			abline(v=3, lty=3)
			abline(v=6, lty=3)
			# loop over narratives
			for(i in 1:length(gs.all))
			{	xs <- sapply(gs.all[[i]], function(g) graph_attr(g,"timestamp"))
				# loop over variables
				for(v in 1:length(variables))
				{	ys <- list.vals[[i]][[v]]
					if(!CUMULATIVE && WINDOW_SIZE>0)
						ys <- sapply(1:length(ys), function(j) mean(ys[max(1,round(j-WINDOW_SIZE/2)):min(length(ys),round(j+WINDOW_SIZE/2))]))
					lines(x=xs, y=ys, col=colors[i], lwd=2, lty=v)
				}
			}
			# add legend
#			legend(
#				x="bottomright",
#				legend=c(narr.names[g.names],"Male","Female"),
#				fill=c(colors,rep(NA,length(variables))), 
#				col=c(rep(NA,length(gs.all)),rep("BLACK",length(variables))), 
#				border=c(rep("BLACK",length(gs.all)),rep(NA,length(variables))), 
##				x.intersp=c(rep(-1,length(gs.all)),rep(1,length(variables))),
#				lwd=2, lty=c(rep(1,length(gs.all)),1:length(variables)),
#				bg="#FFFFFFCC", xjust=1
#			)
			legend(
				x="bottomright",
				legend=c(narr.names[g.names],"Male","Female"),
				col=c(rep(NA,length(gs.all)),rep("BLACK",length(variables))), 
				lwd=2, lty=c(rep(1,length(gs.all)),1:length(variables)),
				bg="#FFFFFFCC"
			)
			legend(
				x="bottomright",
				legend=c(narr.names[g.names],"Male","Female"),
				fill=c(colors,rep(NA,length(variables))),
				border=c(rep("BLACK",length(gs.all)),rep(NA,length(variables))), 
				bg=NA, bty="n"
			)
			dev.off()
	}
}
