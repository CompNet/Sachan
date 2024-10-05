# Sachan

*Story Adaptation and Character Networks*

* Copyright 2023-2024 Arthur Amalvy, Madeleine Janickyj, Shane Mannion, Pádraig MacCarron, and Vincent Labatut 

Sachan is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. For source availability and license information see `licence.txt`

* Lab site: http://lia.univ-avignon.fr/
* GitHub repo: https://github.com/CompNet/Sachan
* Data: https:doi.org/10.5281/zenodo.13893061
* Contacts: Vincent Labatut <vincent.labatut@univ-avignon.fr>, Pádraig MacCarron <padraig.maccarron@ul.ie>


-----------------------------------------------------------------------

If you use this source code or the associated dataset, please cite reference [[A'24](#references)].


# Description
This set of `R` and `Python` scripts aims at analyzing character networks extracted from G. R. R. Martin's *A Song of Ice and Fire* novels, and its adaptations into comics and the TV Show *Game of Thrones*. 

The scripts tackle two tasks described in [[A'24](#references)]. The first task is *character matching*, which consists in identifying pairs of vertices representing the same character in two of these networks, based on the graph structure (and some additional information). The second task is *narrative matching*, and consists in identifying pairs of narrative units (chapters, scenes, episodes...) that represent the same chunk of story in two different media.

Some of the scripts also allow to compute various descriptive statistics. Finally, the scripts also include some processing aiming at extracting the networks from the original raw data, and performing some cleaning. However, the clean networks themselves are also directly provided.


# Data
The networks representing all three media are available online on [Zenodo](https:doi.org/10.5281/zenodo.13893061). This collection includes various types of dynamic graphs (instant vs. cumulative), computed using various narrative units: chapters for novels, scenes and chapters for comics, scene, blocks and episodes for the TV show. The Zenodo repository also includes the many files produced by the scripts. 

![StaticNet](/out/visualization/narratives/static_all.jpg)


# Organization
Here are the folders composing the project:
* Folder `in`: data used by the scripts.
  * Folder `comics`: networks related to the comics.
  * Folder `novels`: networks related to the novels.
  * Folder `narrative_matching`: data used for narrative matching.
  * Folder `tvshow`: networks related to the TV show.
* Folder `out`: files produced by the scripts
  * Folder `centrality`: centrality study.
  * Folder `descript`: descriptive analysis.
  * Folder `narrative_matching`: results of the narrative matching task.
  * Folder `vertex_matching`: results of the character matching task.
  * Folder `visualization`: plots of the networks.
* Folder `src`: source code.
  * Folder `common`: functions used in other scripts.
  * Folder `descript`: descriptive analysis.
  * Folder `narrative_matching`: narrative matching methods.
  * Folder `preprocessing`: extraction and cleaning of the networks.
  * Folder `vertex_matching`: character matching methods.
  * Folder `visualization`: graph plotting.


# Installation
To execute the `R` scripts, you first need to install the language and the required packages:

1. Install the [`R` language](https://www.r-project.org/)
2. Download this project from GitHub and unzip.
3. Install the required packages: 
   1. Open the `R` console.
   2. Set the unzipped directory as the working directory, using `setwd("<my directory>")`.
   3. Run the install script `src/_install.R` (that may take a while).

For the `Python` scripts, the process is similar: 

1. Install the [`Python` language](https://www.python.org/) with a version greater or equal than 3.8. On Linux, `Python` can probably be installed using your preferred package manager.
2. Download this project from GitHub and unzip.
3. Install the required packages:
   1. Open a terminal in the project's root directory
   2. _(optional)_ Create and activate a virtual environment: `python -m venv env && source env/bin/activate`
   3. Install the required dependencies: `pip install -r src/requirements.txt`


# Use
In order to apply the `R` scripts:

1. Open the `R` console.
2. Set the current directory as the working directory, using `setwd("<my directory>")`.
3. Run the main script `src/main.R`.

In order to apply the `Python` scripts:

1. Open a terminal
2. _(optional)_ activate your `Python` environment (e.g. `source env/bin/activate`)
3. Navigate to `src/narrative_matching` (`cd src/narrative_matching`)
4. Run the main computation script `src/narrative_matching/compute_all.sh`

See [the Narrative Matching README](./src/narrative_matching/README.md) for more details.

These scripts produce a number of files in folder `out`.


# Dependencies
Tested with `R` version 4.3.2, with the following packages:
* [`cluster`](https://cran.rstudio.com/web/packages/cluster): version 2.1.6.
* [`fmsb`](https://cran.r-project.org/web/packages/fmsb/): version 0.7.6.
* [`igraph`](http://igraph.org/r/) package: version 1.6.0.
* [`iGraphMatch`](https://cran.r-project.org/web/packages/iGraphMatch/) package: version 2.0.3.
* [`latex2exp`](https://cran.r-project.org/web/packages/latex2exp/): version 0.9.6.
* [`plot.matrix`](https://cran.r-project.org/web/packages/plot.matrix) package: version 1.6.2.
* [`scales`](https://cran.r-project.org/web/packages/scales/): version 1.3.0.
* [`SDMTools`](https://cran.rstudio.com/web/packages/SDMTools): version 1.1-221.2.
* [`viridis`](https://cran.r-project.org/web/packages/viridis/) package: version 0.6.4.
* [`XML`](https://cran.r-project.org/web/packages/XML/): version 3.99-0.16.1.

Tested with `Python` version 3.12 with the following packages:
* [`matplotlib`](https://pypi.org/project/matplotlib/): version 3.5.2.
* [`more-itertools`](https://pypi.org/project/more-itertools/): version 10.1.0.
* [`numpy`](https://pypi.org/project/numpy/): version 1.26.4.
* [`pandas`](https://pypi.org/project/pandas/): version 2.2.0.
* [`sentence-transformers`](https://pypi.org/project/sentence-transformers/): version 2.2.2.
* [`SciencePlots`](https://pypi.org/project/SciencePlots/): version 2.1.0.
* [`scikit-learn`](https://pypi.org/project/scikit-learn/): version 1.3.2.
* [`scipy`](https://pypi.org/project/scipy/): version 1.10.1.
* [`tqdm`](https://pypi.org/project/tqdm/): version 4.65.0.


# To-do List
* ...


# References
* **[A'24]** Amalvy, A.; Janickyj, M.; Mannion, S.; MacCarron, P.; Labatut, V. *Interconnected Kingdoms: Comparing 'A Song of Ice and Fire' Crossmedia Adaptations Using Complex Networks*, Social Network Analysis and Mining, 14:199, 2024.  [⟨hal-xxxxxxxx⟩](https://hal.archives-ouvertes.fr/hal-xxxxxxxx) - DOI: [10.1007/s13278-024-01365-z](https://doi.org/10.1007/s13278-024-01365-z)

