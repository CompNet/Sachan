# Just a list of colors.
# 
# Author: Vincent Labatut
# 09/2023
###############################################################################




###############################################################################
ATT_COLORS_SEX <- c(	# retrieved from https://blog.datawrapper.de/gendercolor/
	"Male"="#19A0AA",		# turquoise
	"Female"="#F15F36",		# salmon
	"Mixed"="#730B6D", 		# purple
	"Unknown"="LIGHTGRAY"	# gray
)




#############################################################
# Combines two colors using a weighted sum of their RGB chanels.
#
# col1: first color.
# col2: second color.
# transparency: alpha level of the first color (percent).
#				0 means pure 1st color, 100 is pure 2nd color.
#
# returns: color resulting from the combination.
#############################################################
combine.colors <- function(col1, col2, transparency=50)
{	transp <- transparency/100.0
	
	# convert to RGB triplet
	rgb1 <- col2rgb(col1, alpha=TRUE)
	rgb2 <- col2rgb(col2, alpha=TRUE)
	
	# create new color using specified transparency
	res <- rgb(
			transp*rgb1[1] + (1-transp)*rgb2[1], 
			transp*rgb1[2] + (1-transp)*rgb2[2], 
			transp*rgb1[3] + (1-transp)*rgb2[3],
			max=255,
			alpha=transp*rgb1[4] + (1-transp)*rgb2[4]
	)
	
	return(res)
}




#############################################################
# Receives a solid color and makes it partially transparent by
# adding an alpha channel.
#
# color: original color.
# transparency: alpha level (percent).
#				100 means completely transparent.
#
# returns: partially transparent color.
#############################################################
make.color.transparent <- function(color, transparency=50)
{	# convert to RGB triplet
	rgb.val <- col2rgb(color)
	
	# create new color using specified transparency
	res <- rgb(
			rgb.val[1], rgb.val[2], rgb.val[3],
			max=255,
			alpha=(100-transparency)*255 / 100
	)
	
	return(res)
}
