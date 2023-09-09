# TODO: Add comment
# 
# Author: Vincent Labatut
###############################################################################




###############################################################################
# Computes statistical mode(s).
#
# x: vector to process.
# na.rm: whether to remove NAs or not.
#
# returns: modal value(s) of the vector.
###############################################################################
mode <- function(x, na.rm=FALSE)
{	if(na.rm)
		x = x[!is.na(x)]	
	vals <- unique(x)
	res <- vals[which.max(tabulate(match(x,vals)))]
	return(res)
}


