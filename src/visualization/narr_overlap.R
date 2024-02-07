# Produces an ad hoc plot for the paper, showing the overlap between the narratives.
# 
# Author: Vincent Labatut
# 02/2024
# 
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/visualization/narr_overlap.R")
###############################################################################
source("src/common/colors.R")




data <- matrix(c(5,0,2,0,5,3),ncol=3)
colnames(data) <- c(expression(italic(Novels)),"Comics","TV Show")
cols <- c(rgb(77,175,74,maxColorValue=255), rgb(255,127,0,maxColorValue=255))

plot.file <- "out/visualization/narratives/narr_overlap.pdf"
pdf(paste0(plot.file,".pdf"), width=15, height=4, bg="white")
	par(mar=c(5, 4, 4-2.5, 2+1.05)+0.1)	# margins Bottom Left Top Right
	barplot(
		height=data, 
		legend.text=c("Adaptation","Extrapolation"), 
		horiz=TRUE, names.arg=c(expression(italic("Novels")),expression(italic("Comics")),expression(italic("TV Show"))),
		col=cols, 
		main="Overlap between the narratives", 
		xlab="Novels/Volumes/Seasons", 
		args.legend=list(x="bottomright")
	)
dev.off()
	
