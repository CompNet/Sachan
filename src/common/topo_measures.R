# Scripts containing the calls allowing to compute the topological measures.
# 
# Author: Vincent Labatut
# 02/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/common/topo_measures.R")
###############################################################################
library("igraph")




###############################################################################
# functions that compute the vertex-related toppological measures.
TOPO_MEAS_VERTICES <- list(
	"degree"=list(
		foo=function(g) {degree(graph=g, mode="all", normalized=FALSE)},
		shortname="Deg.",
		fullname="Degree"),
	"strength"=list(
		foo=function(g) {strength(graph=g, mode="all", weights=E(g)$weight)},
		shortname="Str.",
		fullname="Strength"),
	"closeness"=list(
			foo=function(g) {closeness(graph=g, mode="all", weights=rep(1,gsize(g)), normalized=FALSE)},
		shortname="Clos.",
		fullname="Closeness"),
	"w_closeness"=list(
			foo=function(g) {closeness(graph=g, mode="all", weights=1-E(g)$weight+min(E(g)$weight), normalized=FALSE)},			# need to "reverse" the weights, as the measure considers them as distances 
		shortname="W.Clo.",
		fullname="Weighted Closeness"),
	"betweenness"=list(
			foo=function(g) {betweenness(graph=g, directed=FALSE, weights=rep(1,gsize(g)), normalized=FALSE)},
		shortname="Betw.",
		fullname="Betweenness"),
	"w_betweenness"=list(
			foo=function(g) {betweenness(graph=g, directed=FALSE, weights=1-E(g)$weight+min(E(g)$weight), normalized=FALSE)},	# same as for weighted closeness
		shortname="W.Betw.",
		fullname="Weighted Betweenness"),
	"eigenvector"=list(
			foo=function(g) {eigen_centrality(graph=g, directed=FALSE, scale=FALSE, weights=rep(1,gsize(g)))$vector},
		shortname="Eig.",
		fullname="Eigencentrality"),
	"w_eigenvector"=list(
			foo=function(g) {eigen_centrality(graph=g, directed=FALSE, scale=FALSE, weights=E(g)$weight)$vector},
		shortname="W.Eig",
		fullname="Weighted Eigencentrality"),
	"transitivity"=list(
			foo=function(g) {transitivity(graph=g, type="localundirected", weights=NULL, isolates="zero")},
		shortname="Trans.",
		fullname="Local transitivity"),
	"w_transitivity"=list(
		foo=function(g) {transitivity(graph=g, type="weighted", weights=E(g)$weight, isolates="zero")},
		shortname="W.Trans.",
		fullname="Weighted local transitivity")
)




###############################################################################
# functions that compute the edge-related toppological measures.
TOPO_MEAS_EDGES <- list(
	"distance"=list(
		foo=function(g) {distances(graph=g, mode="all", weights=NA)},
		shortname="Dist.",
		fullname="Distance"),
	"w_distance"=list(
		foo=function(g) {distances(graph=g, mode="all", weights=1-E(g)$weight+min(E(g)$weight))},				# same as for weighted closeness
		shortname="W.Dist.",
		fullname="Weighted distance"),
	"edge_betweenness"=list(
		foo=function(g) {edge_betweenness(graph=g, directed=FALSE, weights=NA)},
		shortname="eBetw.",
		fullname="Edge-betweenness"),
	"w_edge_betweenness"=list(
		foo=function(g) {edge_betweenness(graph=g, directed=FALSE, weights=1-E(g)$weight+min(E(g)$weight))},	# same as for weighted closeness
		shortname="W.eBetw.",
		fullname="Weighted Edge-betweenness")
)




###############################################################################
# functions that compute the graph-related toppological measures.
TOPO_MEAS_GRAPH <- list(
	"vertices"=list(
		foo=function(g) {gorder(graph=g)},
		shortname="Nbr.V.",
		fullname="Number of vertices"),
	"edges"=list(
		foo=function(g) {gsize(graph=g)},
		shortname="Nbr.E.",
		fullname="Number of edges"),
	"density"=list(
		foo=function(g) {graph.density(graph=g)},
		shortname="Dens.",
		fullname="Density"),
	"global_transitivity"=list(
		foo=function(g) {transitivity(graph=g, type="globalundirected", weights=NULL, isolates="zero")},
		shortname="gTrans.",
		fullname="Global transitivity"),
	"components"=list(
		foo=function(g) {components(graph=g, mode="weak")$csize},
		shortname="Nbr.Cpnt.",
		fullname="Number of components"),
	"modularity"=list(
		foo=function(g) {cluster_infomap(graph=g, e.weights=NULL, v.weights=NULL, modularity=TRUE)},
		shortname="Mod.",
		fullname="Modularity"),
	"w_modularity"=list(
		foo=function(g) {cluster_infomap(graph=g, e.weights=E(g)$weight, v.weights=NULL, modularity=TRUE)},
		shortname="W.Mod",
		fullname="Weighted Modularity"),
	"communities"=list(
		foo=function(g) {length(communities(cluster_infomap(graph=g, e.weights=NULL, v.weights=NULL, modularity=FALSE)))},
		shortname="Nbr.Com.",
		fullname="Number of communities"),
	"w_communities"=list(
		foo=function(g) {length(communities(cluster_infomap(graph=g, e.weights=E(g)$weight, v.weights=NULL, modularity=FALSE)))},
		shortname="Nbr.W.Com.",
		fullname="Number of weighted communities")
)




###############################################################################
# everything at once
TOPO_MEAS_ALL <- c(TOPO_MEAS_VERTICES, TOPO_MEAS_EDGES, TOPO_MEAS_GRAPH)
TOPO_MEAS_SHORT_NAMES <- sapply(TOPO_MEAS_ALL, function(l) l$shortname)
TOPO_MEAS_LONG_NAMES <- sapply(TOPO_MEAS_ALL, function(l) l$fullname)
