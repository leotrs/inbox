# INBOX

This package provides code to (I)nspect the (N)on-(B)acktracking (O)r
(X)-centrality of nodes in a graph. The Non-Backtracking centrality (or
NB-centrality for short) was proposed by Travis, et al. [1] as an
alternative to eigenvector centrality that is robust to localization. In a
recent work, Torres et al. [2] propose the X-centrality framework, and
define the _X-Non-Backtracking_ (or X-NB) centrality and the _X-degree
centrality_, and apply it to the task of targeted immunization.


# Installation

To install, simply `git clone` this repository, import the `inbox` module
and call the functions therein.  For `inbox` to work correctly you need to
have installed NetworkX, NumPy, SciPy, and pqdict. To run the notebooks you
also need matplotlib.


# Example

A minimal example of how to use `inbox.py`:

```python
import inbox
import networkx as nx

# inbox works on top of NetworkX
graph = nx.karate_club_graph()

# Use aux=True to compute the auxiliary NB-matrix
nbm = inbox.nb_matrix(graph, aux=False)

# Compute different centrality measures
xnb = inbox.x_nb_centrality(graph)
xdeg = inbox.x_degree(graph)

# Different centralities identify different nodes as most influential
max(xnb, key=xnb.get)      # 33
max(xdeg, key=xdeg.get)    # 2

# Immunize using an X-degree-first strategy
inbox.immunize(graph, 5, strategy='xdeg')    # [2, 33, 0, 30, 23]

```

A more extensive example of the functionality provided in `inbox` can be
found in the [example
notebook](https://github.com/leotrs/inbox/blob/master/example.ipynb).


# References

_[1] Martin, Travis, Xiao Zhang, and Mark EJ Newman. "Localization and
centrality in networks." Physical review E 90.5 (2014): 052808.

_[2] Torres, Leo, Kevin Chan, Hanghang Tong, and Tina
Eliassi-Rad. "Node Immunization with Non-backtracking Eigenvalues."
Preprint.  arXiv:2002.12309. 2020.
