# Just a list of colors.
# 
# Author: Vincent Labatut
# 09/2023
###############################################################################




###############################################################################
# Colors used in the plots.
###############################################################################
MAIN_COLOR <- "#3a548c"

CAT_COLORS_8 <- c(							# basic color brewer palette, see http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=9
	rgb(228,26,28,maxColorValue=255),		# red
	rgb(55,126,184,maxColorValue=255),		# blue
	rgb(77,175,74,maxColorValue=255),		# green
	rgb(152,78,163,maxColorValue=255),		# purple
	rgb(255,127,0,maxColorValue=255),		# orange
	rgb(247,129,191,maxColorValue=255),		# pink
	rgb(166,86,40,maxColorValue=255),		# brown
	rgb(255,255,51,maxColorValue=255)		# yellow
)

# variant with more colors
CAT_COLORS_12 <- c(							# manually extended color brewer palette
	rgb(228,26,28,maxColorValue=255),		# red
	rgb(55,126,184,maxColorValue=255),		# light blue
	rgb(113,219,110,maxColorValue=255),		# light green
	rgb(152,78,163,maxColorValue=255),		# purple
	rgb(255,127,0,maxColorValue=255),		# orange
	rgb(166,86,40,maxColorValue=255),		# brown
	rgb(247,129,191,maxColorValue=255),		# pink
	rgb(153,153,153,maxColorValue=255),		# light grey
	rgb(23,89,143,maxColorValue=255),		# dark blue
	rgb(16,125,12,maxColorValue=255),		# dark green
	rgb(30,30,30,maxColorValue=255),		# dark grey
	rgb(255,255,51,maxColorValue=255)		# yellow
)
CAT_COLORS_18 <- c(							# manual extension of the color brewer palette
	rgb(228,26,28,maxColorValue=255),		# red
	rgb(55,126,184,maxColorValue=255),		# light blue
	rgb(113,219,110,maxColorValue=255),		# light green
	rgb(152,78,163,maxColorValue=255),		# purple
	rgb(255,127,0,maxColorValue=255),		# orange
	rgb(166,86,40,maxColorValue=255),		# brown
	rgb(247,129,191,maxColorValue=255),		# pink
	rgb(153,153,153,maxColorValue=255),		# light grey
	rgb(23,89,143,maxColorValue=255),		# dark blue
	rgb(16,125,12,maxColorValue=255),		# dark green
	rgb(30,30,30,maxColorValue=255),		# dark grey
	rgb(255,255,51,maxColorValue=255),		# yellow
	rgb(143,11,13,maxColorValue=255),		# dark red
	rgb(0,255,255,maxColorValue=255),		# cyan
	rgb(14,161,161,maxColorValue=255),		# dark cyan
	rgb(255,187,120,maxColorValue=255),		# light orange
	rgb(0,0,255,maxColorValue=255),			# straight blue
	rgb(0,255,0,maxColorValue=255)			# straight green
)
CAT_COLORS_22 <- c(	# kelly.colors(22) from package Polychrome
	"#f2f3f4", 		# white 
	"#222222", 		# black 
	"#f3c300", 		# yellow 
	"#875692", 		# purple 
	"#f38400", 		# orange 
	"#a1caf1", 		# lightblue 
	"#be0032", 		# red 
	"#c2b280", 		# buff 
	"#848482", 		# gray 
	"#008856", 		# green 
	"#e68fac", 		# purplishpink 
	"#0067a5", 		# blue 
	"#f99379", 		# yellowishpink 
	"#604e97", 		# violet 
	"#f6a600", 		# orangeyellow 
	"#b3446c", 		# purplishred 
	"#dcd300", 		# greenishyellow 
	"#882d17", 		# reddishbrown 
	"#8db600", 		# yellowgreen 
	"#654522", 		# yellowishbrown 
	"#e25822", 		# reddishorange 
	"#2b3d26"		# olivegreen
)
CAT_COLORS_26 <- c(	# green.armytage.colors(26) from package Polychrome
	"#F0A3FF",		# amethyst 
	"#0075DC",		# blue
	"#993F00",		# caramel
	"#4C005C",		# damson
	"#191919",		# ebony
	"#005C31",		# forest
	"#2BCE48",		# green
	"#FFCC99", 		# honeydew
	"#808080",		# iron
	"#94FFB5",		# jade
	"#8F7C00",		# khaki
	"#9DCC00",		# lime
	"#C20088",		# mallow
	"#003380",		# navy
	"#19A405",		# orpiment
	"#FFA8BB", 		# pink
	"#426600",		# quagmire 
	"#FF0010",		# red
	"#5EF1F2",		# sky
	"#00998F",		# turquoise
	"#E0FF66",		# uranium
	"#100AFF",		# violet
	"#990000",		# wine
	"#FFFF80", 		# xanthin
	"#FFE100",		# yellow
	"#FF5000"		# zinnia
)
CAT_COLORS_32 <- c(	# glasbey.colors(32) from package Polychrome
	"#FFFFFF", "#0000FF", "#FF0000", "#00FF00", "#000033", "#FF00B6", "#005300", "#FFD300", 
	"#009FFF", "#9A4D42", "#00FFBE", "#783FC1", "#1F9698", "#FFACFD", "#B1CC71", "#F1085C", 
	"#FE8F42", "#DD00FF", "#201A01", "#720055", "#766C95", "#02AD24", "#C8FF00", "#886C00", 
	"#FFB79F", "#858567", "#A10300", "#14F9FF", "#00479E", "#DC5E93", "#93D4FF", "#004CFF"
)




#############################################################
# Returns the appropriate number of colors
# 
# values: number of distinct values to plot.
#
# returns: an appropriate palette, for categorical values.
#############################################################
get.palette <- function(values)
{	if(values<=8)
		result <- CAT_COLORS_8
	else if(values<=12)
		result <- CAT_COLORS_12
	else if(values<=18)
		result <- CAT_COLORS_18
	else if(values<=22)
		result <- CAT_COLORS_22
	else if(values<=26)
		result <- CAT_COLORS_26
	else
		result <- CAT_COLORS_32
	
	result <- result[1:values]
	return(result)
}




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




###############################################################################
ATT_COLORS_SEX <- c(		# retrieved from https://blog.datawrapper.de/gendercolor/
	# single
	"Male"="#19A0AA",		# turquoise
	"Female"="#F15F36",		# salmon
	"Mixed"="#730B6D", 		# purple
	"Unknown"="LIGHTGRAY",	# gray
	# double
	"Female-Female"="#F15F36",
	"Female-Male"=combine.colors("#19A0AA", "#F15F36", transparency=50),
	"Male-Male"="#19A0AA",
	# triple
	"Female-Female-Female"="#F15F36",
	"Female-Female-Male"=combine.colors("#19A0AA", "#F15F36", transparency=33),
	"Female-Male-Male"=combine.colors("#19A0AA", "#F15F36", transparency=67),
	"Male-Male-Male"="#19A0AA"
)
ATT_COLORS_BOOL <- c(
	"FALSE"=rgb(255,127,0,maxColorValue=255),		# orange
	"TRUE"=rgb(77,175,74,maxColorValue=255)			# green
)

# overall list
ATT_COLORS <- list(
	Named=ATT_COLORS_BOOL,
	Sex=ATT_COLORS_SEX
)
