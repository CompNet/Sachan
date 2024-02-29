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
		type="vertex",
		foo=function(g) {degree(graph=g, mode="all", normalized=FALSE)},
		shortname="Deg.",
		fullname="Degree"),
	"strength"=list(
		type="vertex",
		foo=function(g) {strength(graph=g, mode="all", weights=E(g)$weight)},
		shortname="Str.",
		fullname="Strength"),
	"closeness"=list(
		type="vertex",
		foo=function(g) {closeness(graph=g, mode="all", weights=rep(1,gsize(g)), normalized=FALSE)},
		shortname="Clos.",
		fullname="Closeness"),
	"w_closeness"=list(
		type="vertex",
		foo=function(g) {closeness(graph=g, mode="all", weights=1-E(g)$weight+min(E(g)$weight), normalized=FALSE)},			# need to "reverse" the weights, as the measure considers them as distances 
		shortname="W.Clo.",
		fullname="Weighted Closeness"),
	"betweenness"=list(
		type="vertex",
		foo=function(g) {betweenness(graph=g, directed=FALSE, weights=rep(1,gsize(g)), normalized=FALSE)},
		shortname="Betw.",
		fullname="Betweenness"),
	"w_betweenness"=list(
		type="vertex",
		foo=function(g) {betweenness(graph=g, directed=FALSE, weights=1-E(g)$weight+min(E(g)$weight), normalized=FALSE)},	# same as for weighted closeness
		shortname="W.Betw.",
		fullname="Weighted Betweenness"),
	"eigenvector"=list(
		type="vertex",
		foo=function(g) {eigen_centrality(graph=g, directed=FALSE, scale=FALSE, weights=rep(1,gsize(g)))$vector},
		shortname="Eig.",
		fullname="Eigencentrality"),
	"w_eigenvector"=list(
		type="vertex",
		foo=function(g) {eigen_centrality(graph=g, directed=FALSE, scale=FALSE, weights=E(g)$weight)$vector},
		shortname="W.Eig",
		fullname="Weighted Eigencentrality"),
	"transitivity"=list(
		type="vertex",
		foo=function(g) {transitivity(graph=g, type="localundirected", weights=NULL, isolates="zero")},
		shortname="Trans.",
		fullname="Local Transitivity"),
	"w_transitivity"=list(
		type="vertex",
		foo=function(g) {transitivity(graph=g, type="weighted", weights=E(g)$weight, isolates="zero")},
		shortname="W.Trans.",
		fullname="Weighted Local Transitivity")
)




###############################################################################
# functions that compute the edge-related toppological measures.
TOPO_MEAS_EDGES <- list(
	"distance"=list(
		type="edge",
		foo=function(g) {m <- distances(graph=g, mode="all", weights=NA); m[upper.tri(m,diag=FALSE)]},
		shortname="Dist.",
		fullname="Distance"),
	"w_distance"=list(
		type="edge",
		foo=function(g) {m <- distances(graph=g, mode="all", weights=1-E(g)$weight+min(E(g)$weight)); m[upper.tri(m,diag=FALSE)]},	# same as for weighted closeness
		shortname="W.Dist.",
		fullname="Weighted Distance"),
	"edge_betweenness"=list(
		type="edge",
		foo=function(g) {edge_betweenness(graph=g, directed=FALSE, weights=NA)},
		shortname="eBetw.",
		fullname="Edge-Betweenness"),
	"w_edge_betweenness"=list(
		type="edge",
		foo=function(g) {edge_betweenness(graph=g, directed=FALSE, weights=1-E(g)$weight+min(E(g)$weight))},						# same as for weighted closeness
		shortname="W.eBetw.",
		fullname="Weighted Edge-Betweenness")
)




###############################################################################
# functions that compute the graph-related toppological measures.
TOPO_MEAS_GRAPH <- list(
	"vertices"=list(
		type="graph",
		foo=function(g) {gorder(graph=g)},
		shortname="Nbr.V.",
		fullname="Number of Vertices"),
	"edges"=list(
		type="graph",
		foo=function(g) {gsize(graph=g)},
		shortname="Nbr.E.",
		fullname="Number of Edges"),
	"weights"=list(
		type="graph",
		foo=function(g) {sum(E(g)$weight)},
		shortname="Sum Wgt.",
		fullname="Sum of Weights"),
	"density"=list(
		type="graph",
		foo=function(g) {graph.density(graph=g)},
		shortname="Dens.",
		fullname="Density"),
	"global_transitivity"=list(
		type="graph",
		foo=function(g) {transitivity(graph=g, type="globalundirected", weights=NULL, isolates="zero")},
		shortname="gTrans.",
		fullname="Global Transitivity"),
	"components"=list(
		type="graph",
		foo=function(g) {components(graph=g, mode="weak")$no},
		shortname="Nbr.Cpnt.",
		fullname="Number of Components"),
	"modularity"=list(
		type="graph",
		foo=function(g) {modularity(cluster_infomap(graph=g, e.weights=NULL, v.weights=NULL, modularity=TRUE))},
		shortname="Mod.",
		fullname="Modularity"),
	"w_modularity"=list(
		type="graph",
		foo=function(g) {modularity(cluster_infomap(graph=g, e.weights=E(g)$weight, v.weights=NULL, modularity=TRUE))},
		shortname="W.Mod",
		fullname="Weighted Modularity"),
	"communities"=list(
		type="graph",
		foo=function(g) {length(communities(cluster_infomap(graph=g, e.weights=NULL, v.weights=NULL, modularity=FALSE)))},
		shortname="Nbr.Com.",
		fullname="Number of Communities"),
	"w_communities"=list(
		type="graph",
		foo=function(g) {length(communities(cluster_infomap(graph=g, e.weights=E(g)$weight, v.weights=NULL, modularity=FALSE)))},
		shortname="Nbr.W.Com.",
		fullname="Number of Weighted Communities")
)




###############################################################################
# everything at once
TOPO_MEAS_ALL <- c(TOPO_MEAS_VERTICES, TOPO_MEAS_EDGES, TOPO_MEAS_GRAPH)
TOPO_MEAS_SHORT_NAMES <- sapply(TOPO_MEAS_ALL, function(l) l$shortname)
TOPO_MEAS_LONG_NAMES <- sapply(TOPO_MEAS_ALL, function(l) l$fullname)
