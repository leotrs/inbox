"""
inbox.py
--------

This file contains utilities to compute non-backtracking centrality,
X-non-backtracking centrality, X-degree centrality, and related
computations.

"""

import numpy as np
import networkx as nx
import scipy.sparse as sparse
from collections import defaultdict
from pqdict import maxpq


######################################################################
###                            Matrices                            ###
######################################################################

def half_incidence(graph, ordering='blocks', return_ordering=False):
    """Return the 'half incidence' matrices of the graph.

    test

    The resulting matrices have shape of (n, 2m), where n is the number of
    nodes and m is the number of edges.


    Params
    ------

    graph (nx.Graph): The graph.

    ordering (str): If 'blocks' (default), the two columns corresponding to
    the i'th edge are placed at i and i+m. That is, choose an arbitarry
    direction for each edge in the graph. The first m columns correspond to
    this orientation, while the latter m columns correspond to the reversed
    orientation. Columns in both blocks are sorted following graph.edges.
    If 'consecutive', the first two columns correspond to the two
    orientations of the first edge, the third and fourth row are the two
    orientations of the second edge, and so on. In general, the two columns
    for the i'th edge are placed at 2i and 2i+1. If 'custom', parameter
    custom must be a dictionary of pairs of the form (idx, (i, j)) where
    the key idx maps onto a 2-tuple of indices where the edge must fall.

    custom (dict): Used only when ordering is 'custom'.

    return_ordering (bool): If True, return a function that maps an edge id
    to the column placement. That is, if ordering=='blocks', return the
    function lambda x: (x, m+x), if ordering=='consecutive', return the
    function lambda x: (2*x, 2*x + 1). If False, return None.


    Returns
    -------

    (source, target), or (source, target, ord_function) if return_ordering
    is True.


    Notes
    -----

    Assumes the nodes are labeled by consecutive integers starting at 0.

    """
    numnodes = graph.order()
    numedges = graph.size()

    if ordering == 'blocks':
        def src_pairs(i, u, v): return [(u, i), (v, numedges + i)]
        def tgt_pairs(i, u, v): return [(v, i), (u, numedges + i)]
    if ordering == 'consecutive':
        def src_pairs(i, u, v): return [(u, 2*i), (v, 2*i + 1)]
        def tgt_pairs(i, u, v): return [(v, 2*i), (u, 2*i + 1)]
    if isinstance(ordering, dict):
        def src_pairs(i, u, v): return [
            (u, ordering[i][0]), (v, ordering[i][1])]
        def tgt_pairs(i, u, v): return [
            (v, ordering[i][0]), (u, ordering[i][1])]

    def make_coo(make_pairs):
        """Make a sparse 0-1 matrix.

        The returned matrix has a positive entry at each coordinate pair
        returned by make_pairs, for all (idx, node1, node2) edge triples.

        """
        coords = [pair
                  for idx, (node1, node2) in enumerate(graph.edges())
                  for pair in make_pairs(idx, node1, node2)]
        data = np.ones(len(coords))
        return sparse.coo_matrix((data, list(zip(*coords))),
                                 shape=(numnodes, 2*numedges))

    source = make_coo(src_pairs).asformat('csr')
    target = make_coo(tgt_pairs).asformat('csr')

    if return_ordering:
        if ordering == 'blocks':
            def ord_func(x): return (x, numedges+x)
        elif ordering == 'consecutive':
            def ord_func(x): return (2*x, 2*x+1)
        elif isinstance(ordering, dict):
            def ord_func(x): return ordering[x]
        return source, target, ord_func
    else:
        return source, target


def nb_matrix(graph, aux=False, ordering='blocks', return_ordering=False):
    """Return NB-matrix of a graph.

    If aux=False, return the true non-backtracking matrix, defined as the
    unnormalized transition matrix of a random walker that does not
    backtrack. If the graph has m edges, the NB-matrix is 2m x 2m. The rows
    and columns are ordered according to ordering (see half_incidence).

    If aux=True, return the auxiliary NB-matrix of a graph is the block
    matrix defined as

    B' = [0  D-I]
         [-I  A ]

    Where D is the degree-diagonal matrix, I is the identity matrix and A
    is the adjacency matrix. If the graph has n nodes, the auxiliary
    NB-matrix is 2n x 2n. The rows and columns are sorted in the order of
    the nodes in the graph object.

    Params
    ------

    graph (nx.Graph): the graph.

    aux (bool): whether to return the auxiliary or the true NB-matrix.

    ordering ('blocks' or 'consecutive'): ordering of the rows and columns
    if aux=False (see half_incidence). If aux=True, the rows and columns of
    the result will always be in accordance to the order of the nodes in
    the graph, regardless of the value of ordering.

    return_ordering (bool): if True, return the edge ordering used (see
    half_incidence).

    Returns
    -------

    matrix (scipy.sparse): (auxiliary) NB-matrix in sparse CSR format.

    matrix, ordering_func: if return_ordering=True.

    """
    if aux:
        degrees = graph.degree()
        degrees = sparse.diags([degrees[n] for n in graph.nodes()])
        ident = sparse.eye(graph.order())
        adj = nx.adjacency_matrix(graph)
        pseudo = sparse.bmat([[None, degrees - ident], [-ident, adj]])
        return pseudo.asformat('csr')

    else:
        # Compute the NB-matrix in a single pass on the non-zero elements
        # of the intermediate matrix.
        sources, targets, ord_func = half_incidence(
            graph, ordering, return_ordering=True)
        inter = np.dot(sources.T, targets).asformat('coo')
        inter_coords = set(zip(inter.row, inter.col))

        # h_coords contains the (row, col) coordinates of non-zero elements
        # in the NB-matrix
        h_coords = [(r, c)
                    for r, c in inter_coords if (c, r) not in inter_coords]
        data = np.ones(len(h_coords))
        nbm = sparse.coo_matrix((data, list(zip(*h_coords))),
                                shape=(2*graph.size(), 2*graph.size()))

        # Return the correct format
        nbm = nbm.asformat('csr')
        return (nbm, ord_func) if return_ordering else nbm


def perm_matrix(size):
    """Return the matrix P such that BP is symmetric.

    Params
    ------

    size (int): the number of edges in the graph.

    Returns
    -------

    matrix (scipy.sparse): square matrix of size 2*size.

    Notes
    -----

    This matrix only works when the NB-matrix B has been built by using the
    'blocks' ordering.

    """
    return sparse.bmat([[None, sparse.identity(size)],
                        [sparse.identity(size), None]])


def x_matrix(graph, remove_node=None, add_neighbors=None, return_all=False):
    """Compute the X matrix corresponding to node addition or removal.

    The arguments remove_node and add_neighbors are mutually exclusive, one
    of them must always be None. Assume the graph has m edges and its
    NB-matrix is B.


    If remove_node is given, it must be a node in the graph. Suppose its
    degree is d. In this case, partition the NB-matrix of graph as

    B = [B' D]
        [E  F]

    where D is (2m - 2d) x 2d, E is 2d x (2m - 2d), and F is 2d x 2d.
    Here, B' is the NB-matrix after node removal.

    If add_neighbors is given, it must be a list of nodes to which the new
    node will connect to. The NB-matrix of the graph after addition is
    given by

        [B D]
        [E F]

    Let d = len(neighbors). Here, B is the NB-matrix of the original graph
    (before addition), D is 2m x 2d, E is 2d x 2m, and F is 2d x 2d.

    In either of the above cases, X is always X = D F E.

    Note that there are two ways of getting the NB-matrix of graph after
    removing node:

    ```
    # first way
    >>> graph.remove(node)
    >>> nbm = nb_matrix(graph, aux=False)

    # second way
    >>> nbm, D, E, F = X_matrix(graph, remove_node=node, return_all=True)
    ```

    The main difference between these two is that the rows and columns will
    be sorted differently, and it will be error-prone to get the matrices,
    D, E, F when using the first way.


    Params
    ------

    graph (nx.Graph): the graph.

    remove_node (node label): if removing, the node to remove.

    add_neighbors (sequence of node label): if adding, the nodes that will
    become neighbors of the new node.

    return_all (bool): whether to return the four blocks (True), or just
    the matrix X (False, default).


    Returns
    -------

    (X,) if return_all is False, or (B, D, E, F) or return_all is True. All
    are scipy.sparse matrices. In the latter case, one can compute X by
    multiplying D*F*E.


    Notes
    -----

    The graph is assumed to have nodes labeled by consecutive integers
    starting at zero. The graph is never manipulated, i.e. a node is never
    added to or removed from it. For that, use graph.add_node or
    graph.remove_node.

    """
    if remove_node is not None:
        nbm, ordering = nb_matrix(graph, return_ordering=True)
        incident = np.array(
            [ordering(i) for i, edge in enumerate(graph.edges())
             if remove_node in edge]).ravel()
        not_incident = set(range(2 * graph.size())) - set(incident)
        not_incident = np.array(list(not_incident))
        B = nbm[not_incident, :][:, not_incident]
        D = nbm[not_incident, :][:, incident]
        E = nbm[incident, :][:, not_incident]
        F = nbm[incident, :][:, incident]
        return (B, D, E, F) if return_all else D.dot(F.dot(E))

    else:
        # Handle dummy case
        degree = len(add_neighbors)
        if degree == 0:
            B = nb_matrix(graph)
            D = sparse.csr_matrix(np.empty((0, 0)))
            E = sparse.csr_matrix(np.empty((0, 0)))
            F = sparse.csr_matrix(np.empty((0, 0)))
            return (B, D, E, F) if return_all else D.dot(F.dot(E))

        # Create a new graph so it is easy to compute its NB-matrix
        new_node = graph.order()
        new_graph = graph.copy()
        new_edges = [(n, new_node) for n in add_neighbors]
        new_graph.add_edges_from(new_edges)

        # If no custom ordering is specified, use the one used when
        # building the NB-matrix on the original graph
        nbm, ordering = nb_matrix(graph, return_ordering=True)
        ordering = {i: ordering(i) for i in range(graph.size())}

        # Set up a custom ordering that puts the original edges first and
        # the new edges last
        size = graph.size()
        prev_idx = {e: i for i, e in enumerate(graph.edges())}
        custom = {}
        count = 0
        for idx, edge in enumerate(new_graph.edges()):
            if graph.has_edge(*edge):
                prev = prev_idx.get(edge, prev_idx.get((edge[1], edge[0])))
                custom[idx] = ordering[prev]
            else:
                custom[idx] = (2*size + 2*count, 2*size + 2*count + 1)
                count += 1

        # Using the custom ordering, get the correct blocks
        new_nbm = nb_matrix(new_graph, ordering=custom)
        old_edges = 2 * size
        new_edges = 2 * degree
        B = new_nbm[:old_edges, :][:, :old_edges]
        D = new_nbm[:old_edges, :][:, -new_edges:]
        E = new_nbm[-new_edges:, :][:, :old_edges]
        F = new_nbm[-new_edges:, :][:, -new_edges:]
        return (B, D, E, F) if return_all else D.dot(F.dot(E))


######################################################################
###                           Centrality                           ###
######################################################################

def nb_centrality(graph, normalized=True, return_eigenvalue=False, tol=0):
    """Return the non-backtracking centrality of each node.

    The nodes must be labeled by consecutive integers starting at zero.

    Params
    ------

    graph (nx.Graph): the graph.

    normalized (bool): whether to return the normalized version,
    corresponding to v^T P v = 1.

    return_eigenvalue (bool): whether to return the largest
    non-backtracking eigenvalue as part of the result.

    tol (float): the tolerance for eignevecto computation. tol=0 (default)
    means machine precision.

    Returns
    -------

    centralities (dict): dictionary of {node: centrality} items.

    centralities (dict), eigenvalue (float): if return_eigenvalue is True.

    """
    # Matrix computations require node labels to be consecutive integers,
    # so we need to (i) convert them if they are not, and (ii) preserve the
    # original labels as an attribute.
    graph = nx.convert_node_labels_to_integers(
        graph, label_attribute='original_label')

    # The centrality is given by the first entries of the principal left
    # eigenvector of the auxiliary NB-matrix
    val, vec = sparse.linalg.eigs(nb_matrix(graph, aux=True).T, k=1, tol=tol)
    val = val[0].real
    vec = vec[:graph.order()].ravel()

    # Sometimes the vector is returned with all negative components and we
    # need to flip the sign.  To check for the sign, we check the sign of
    # the sum, since every element has the same sign (or is zero).
    if vec.sum() < 0:
        vec *= -1

    # Currently, vec is unit length. The 'correct' normalization requires
    # that we scale it by \mu.
    if normalized:
        vec *= compute_mu(graph, val, vec)

    # Pack everything in a dict and return
    result = {graph.nodes[n]['original_label']: vec[n].real for n in graph}
    return (result, val) if return_eigenvalue else result


def compute_mu(graph, val, vec):
    """Compute mu given the leading eigenpair.

    Params
    ------

    graph (nx.Graph): the graph.

    val (float): the leading eigenvalue.

    vec (np.array): the first half of the principal left unit eigenvector.

    Returns
    -------

    mu (float): a constant such that mu * vec is the 'correctly' normalized
    non-backtracking centrality.

    """
    degs = graph.degree()
    coef = sum(vec[n]**2 * degs(n) for n in graph)
    return np.sqrt(val * (val**2 - 1) / (1 - coef))


def x_nb_centrality(graph, approx=True, return_eigenvalue=False, tol=0):
    """Return the X-NB centrality of each node.

    Params
    ------

    graph (nx.Graph): the graph.

    approx (bool): if True (default), the NB-centrality of each node is
    computed in the original graph, without removing any nodes.

    return_eigenvalue (bool): whether to return the largest
    non-backtracking eigenvalue as part of the result.

    tol (float): the tolerance for eignevecto computation. tol=0 (default)
    means machine precision.

    Returns
    -------

    centralities (dict): dictionary of {node: centrality} items.

    centralities (dict), eigenvalue (float): if return_eigenvalue is True.

    """
    if approx:
        # Get the correctly normalized NB-centralities
        nb_cent, val = nb_centrality(
            graph, return_eigenvalue=True, normalized=True, tol=tol)

        # Aggregate each node's neighbor's NB-centralities
        nb_cent = np.array([nb_cent[n] for n in graph])
        adj = nx.to_scipy_sparse_matrix(graph)
        xnb_cent = adj.dot(nb_cent)**2 - adj.dot(nb_cent**2)

    else:
        # We can compute the true value in one of two ways: (i) remove each
        # node from the graph, compute the NB-centralities of all nodes in
        # the resulting graph, and aggregate them to get the X-NB
        # centralities; or we could (ii) compute \alpha directly by using
        # the X matrix. Here we use the second method.
        xnb_cent = []
        for node in graph:
            B, D, E, F = x_matrix(graph, remove_node=node)
            X = D.dot(F).dot(E)
            val, right = sparse.linalg.eigs(B, k=1, tol=tol)
            right = right.ravel().real
            left = perm_matrix(B.shape[0] // 2).dot(right)
            result = left.dot(X.dot(right)) / left.dot(right)
            xnb_cent.append(result)

    # This handles both consecutive integer and arbitrary labels
    result = {n: xnb_cent[i].real for i, n in enumerate(graph)}
    return (result, val) if return_eigenvalue else result


def x_degree(graph):
    """Return the X-degree centrality of each node.

    Params
    ------

    graph (nx.Graph): the graph.

    Returns
    -------

    centralities (dict): dictionary of {node: centrality} items.

    """
    degrees = graph.degree()
    def agg(arr): return arr.sum()**2 - (arr**2).sum()
    return {node: agg(np.array([degrees[n] - 1 for n in graph.neighbors(node)]))
            for node in graph}


def collective_influence(graph):
    """Return the collective influence of each node.

    Here we only use the immediate neighborhood of each node to compute its
    collective influence, i.e. we use l = 1 in equation (1) of [1].

    Params
    ------

    graph (nx.Graph): the graph.

    Returns
    -------

    centralities (dict): dictionary of {node: centrality} items.

    References
    ----------

    [1] Morone, Flaviano, et al. "Collective influence algorithm to find
    influencers via optimal percolation in massively large social media."
    Scientific reports 6 (2016): 30062.

    """
    deg = graph.degree()
    return {node: (deg[node] - 1) * sum(deg[n] - 1 for n in graph.neighbors(node))
            for node in graph}


def x_centrality(graph, values):
    """X-centrality from a vector of values for each directed edge.

    for x-deg do not call this

    values must be in 'blocks' ordering

    """
    # First aggregation: sum all values of edges pointing toward a node
    _, target = half_incidence(graph)
    result = target.dot(values)

    # Second aggregation: square of the sum minus sum of squares of
    # neighbors
    adj = nx.to_scipy_sparse_matrix(graph)
    xnb_centrality = adj.dot(result)**2 - adj.dot(result**2)

    # Pack in a dict and return
    return {n: xnb_centrality[i].real for i, n in enumerate(graph)}


def shave(graph):
    """Return the 2-core of a graph.

    Iteratively remove the nodes of degree 0 or 1, until all nodes have
    degree at least 2.

    """
    core = graph.copy()
    while True:
        to_remove = [node for node, neighbors in core.adj.items()
                     if len(neighbors) < 2]
        core.remove_nodes_from(to_remove)
        if not to_remove:
            break
    return core


######################################################################
###                          Immunization                          ###
######################################################################

def immunize(graph, num_nodes, strategy='xdeg', queue=True, min_deg=2,
             tol=0):
    """Remove num_nodes from the graph according to the strategy.

    Targeted immunization works by (i) computing a score of each node, (ii)
    removing the node with the highest score, and (iii) iterating until the
    target number of nodes has been removed. Importantly, the score has to
    be recomputed at each step.

    Params
    ------

    graph (nx.Graph): the graph to immunize.

    num_nodes (int): the number of nodes to remove.

    strategy (str): the strategy to use to compute the score. Can be one of
    the following:
      - 'xdeg' (default): X-degree.
      - 'xnb': the approximate version of X-non-backtracking centrality.
      - 'nb': non-backtracking centrality.
      - 'ci': collective influence.
      - 'deg': node degree.
      - 'ns': NetShield.
      - 'core': k-core index.

    queue (bool): for 'xdeg', 'ci', or 'deg', the algorithm may use either
    an indexed priority queue (when queue=True, default) or a dictionary
    (when queue=False) to store the score values. The dictionary version
    has better worst case scenario complexity, though the queue version
    seems to be faster in practice for a variety of graphs. For all other
    strategies, this is ignored.

    min_deg (int): the minimum degree of nodes considered for
    immunization. 'xdeg', 'xnb', 'nb', and 'ci' all assign zero score to
    nodes of degree 1, and thus these are never chosen for immunization. It
    is therefore recommended to use min_deg=2 (default) when using said
    strategies; this does not change the outcome of immunization. For other
    strategies, this may change the outcome. In all cases, the algorithm
    will run faster the larger the value of min_deg as fewer nodes will be
    considered. Using min_deg>2 will always be faster, though it may not
    yield theoretically optimal results.

    tol (float): for 'nb' or 'xnb', numerical tolerance for eigenvector
    computation. Ignored in all oter strategies. If tol=0 (default), use
    machine precision.


    Returns
    -------

    (nodes, new_graph): nodes is a list of the removed nodes, in order of
    removal (the zeroth element was removed first), new_graph is the graph
    after immuniaztion.


    Notes
    -----

    Tie breaks are decided arbitrarily. In particular, the two versions of
    the strategies 'deg', 'ci', 'xdeg' (i.e. using an indexed priority
    queue or a map) may yield different results as the underlying data
    structures may break ties differently.

    """
    # Run the immunization on a reduced graph, so it is faster
    reduced = graph
    if min_deg > 0:
        deg = reduced.degree()
        reduced = reduced.subgraph(n for n in reduced if deg[n] >= min_deg)

    # Remember to run the immunization on the reduced graph...
    if strategy == 'xdeg':
        adj = {n: set(reduced.neighbors(n)) for n in reduced}
        nodes = _immunize_xdeg(adj, num_nodes, queue)
    elif strategy == 'ci':
        adj = {n: set(reduced.neighbors(n)) for n in reduced}
        nodes = _immunize_ci(adj, num_nodes, queue)
    elif strategy == 'deg':
        adj = {n: set(reduced.neighbors(n)) for n in reduced}
        nodes = _immunize_deg(adj, num_nodes, queue)
    elif strategy == 'ns':
        nodes = _immunize_netshield(reduced.copy(), num_nodes)
    else:
        score_func = {
            'core': nx.core_number,
            'nb': lambda g: nb_centrality(g, normalized=False, tol=tol),
            'xnb': lambda g: x_nb_centrality(g, tol=tol),
        }[strategy]

        # Remember to run the immunization on the reduced graph...
        reduced = reduced.copy()
        nodes = []
        for _ in range(num_nodes):
            score = score_func(reduced)
            next_node = max(score, key=score.get)
            reduced.remove_node(next_node)
            nodes.append(next_node)

    # Return a subgraph view containing all those nodes of small degree
    nodes_set = set(nodes)
    new_graph = graph.subgraph(n for n in reduced if n not in nodes_set)
    return nodes, new_graph


def _immunize_deg(adj, num_nodes, queue=True):
    """Internal function. To immunize a graph, use `immunize`."""
    # Takes O(n) time and O(n) space
    deg = {n: len(adj[n]) for n in adj}

    # Make sure we don't remove more nodes than there are available
    num_nodes = min(num_nodes, len(deg))

    # If using a queue, we need to heapify it (which takes O(n))
    if queue:
        deg = maxpq(deg)

    # Main loop
    removed = []
    for _ in range(num_nodes):
        if queue:
            node = deg.pop()             # Takes O(log n)
        else:
            node = max(deg, key=deg.get)  # Takes O(n)
            del deg[node]

        # Takes O(degree[node])
        for neigh in adj[node]:
            adj[neigh].remove(node)
            deg[neigh] -= 1

        # Finally, finish udpating the graph, and store the node.
        del adj[node]
        removed.append(node)

    return removed


def _immunize_xdeg(adj, num_nodes, queue=True):
    """Internal function. To immunize a graph, use `immunize`."""
    # Takes O(m) time and O(2n) space
    to_be_squared, sum_squares = defaultdict(int), defaultdict(int)
    for node in adj:
        for neigh in adj[node]:
            # Note we keep the sum of squares, not its square, and square
            # it only when needed.
            to_be_squared[node] += len(adj[neigh]) - 1
            sum_squares[node] += (len(adj[neigh]) - 1)**2

    # Takes O(n) time and O(n) space
    # Remember to square the first term
    xdeg = {n: to_be_squared[n]**2 - sum_squares[n] for n in adj}

    # We actually don't need this again
    del sum_squares

    # Make sure we don't remove more nodes than there are available
    num_nodes = min(num_nodes, len(xdeg))

    # If using a queue, we need to heapify it (which takes O(n))
    if queue:
        xdeg = maxpq(xdeg)

    # Main loop
    # Takes O(m) time and O(m) space
    removed = []
    for _ in range(num_nodes):
        if queue:
            node = xdeg.pop()             # Takes O(log n)
        else:
            node = max(xdeg, key=xdeg.get)  # Takes O(n)
            del xdeg[node]

        # Takes O(degree[node])
        for neigh in adj[node]:
            adj[neigh].remove(node)

        # The following loop will compute the difference in xdeg of each
        # node, without changing any of the variables. The loop after that
        # will actually apply the changes.
        #
        # We do this as follows. For a node i, define s(i) to be the sum of
        # the excess degrees of its neighbors, i.e. s(i) ==
        # to_be_squared[i]. Let deg be the degree of the target node. There
        # are four types of nodes that will be affected by the removal:
        #
        # 1. The nodes that are 1hop neighbors but not 2hop heighbors of
        # the target node will have their degree decrase by 1, and their
        # xdeg decreased by 2(s(i) - deg + 1)(deg - 1).
        #
        # 2. The nodes that are 2hop neighbors but not 1hop neighbors of
        # the target node will have their degree decreased by t(i), where
        # t(i) is the number of common neighbors they share with the target
        # node (i.e. the number of paths of length 2 between the target
        # node and i). Their xdeg will decrease by 2t(i)s(i) + t(i) -
        # t(i)**2 - 2p(i). Here, p(i) is the sum of the excess degrees of
        # the neighbors of i who are also neighbors of the target node.
        #
        # 3. The nodes that are both 1hop and 2hop neighbors will have
        # their degree decrease by t(i) + deg - 1. Their xdeg will decrease
        # by 2t(i)s(i) + t(i) - t(i)**2 - 2p(i) + 2(s(i) - deg + 1)(deg -
        # 1) - 2 t(i)(deg - 1). Note this is the sum of the changes for
        # 1hop and 2hop neighbors, plus the additional term 2 t(i)(deg - 1).
        #
        # 4. The target node itself will be removed: its degree and xdeg
        # will decrase to zero.
        deg = len(adj[node])
        delta = defaultdict(int)
        count = defaultdict(int)

        # Compute the deltas. Takes O(degree^2[node])
        for neigh in adj[node]:
            delta[neigh] += \
                (2                                  # 2
                 * (to_be_squared[neigh] - deg + 1)  # (s - deg + 1)
                 * (deg - 1))                       # (deg - 1)
            for neigh2 in adj[neigh]:
                # Each time r we visit a node i through a 2hop path, we are
                # adding 2s(i) + 1 - (2r + 1) - 2(p_r - 1), where p_r is
                # the degree of the node that led us to i. After visiting
                # t(i) times, this adds up to 2t(i)s(i) + t(i) - t(i)**2 -
                # 2p(i), as desired. Note that p_r is the degree BEFORE any
                # changes have been made to the network, but the degrees of
                # the neighbors of the target node already changed in the
                # previous loop, therefore p_r - 1 = len(adj[neigh]).
                delta[neigh2] += \
                    (2 * to_be_squared[neigh2]           # 2s
                     + 1                                 # + 1
                     - (2 * count[neigh2] + 1)           # - (2r + 1)
                     - 2 * len(adj[neigh]))              # - 2(p_r - 1)

                # Increment the count r(i). At the end of this double loop,
                # we will have count[i] == t(i).
                count[neigh2] += 1

        # Apply the changes at the same time. Takes O(degree^2[node]).
        for neigh2 in delta:
            # At the end of the previous loop, nodes of type 1 and 2
            # already have the correct deltas, while nodes of type 3 are
            # missing a term. We can finally update xdeg.
            if neigh2 in adj[node] and count[neigh2] > 0:
                delta[neigh2] -= 2 * count[neigh2] * (deg - 1)

            # If dict, takes O(1). If heap, takes O(log n)
            xdeg[neigh2] = xdeg[neigh2] - delta[neigh2]

            # Update the s(i) of each node. Note this is simply computing
            # the changes in i's neighbors' degrees. Note nodes of type 3
            # receive both updates.
            if neigh2 in adj[node]:
                to_be_squared[neigh2] -= deg - 1
            if count[neigh2] > 0:
                to_be_squared[neigh2] -= count[neigh2]

        # Finally, finish udpating the graph, and store the node.
        del adj[node]
        del to_be_squared[node]
        removed.append(node)

    return removed


def _immunize_ci(adj, num_nodes, queue=True):
    """Internal function. To immunize a graph, use `immunize`."""
    # Once populated, will take O(n) space
    ci = defaultdict(int)

    # Takes O(m) time
    for node in adj:
        excess_deg = len(adj[node]) - 1
        for neigh in adj[node]:
            ci[node] += excess_deg * (len(adj[neigh]) - 1)

    # Make sure we don't remove more nodes than there are available
    num_nodes = min(num_nodes, len(ci))

    # If using a queue, we need to heapify it (which takes O(n))
    if queue:
        ci = maxpq(ci)

    # Main loop
    removed = []
    for _ in range(num_nodes):
        if queue:
            node = ci.pop()  # Takes O(log n)
        else:
            node = max(ci, key=ci.get)  # Takes O(n)
            del ci[node]

        # Takes O(degree[node])
        for neigh in adj[node]:
            adj[neigh].remove(node)

        # Compute the deltas. Takes O(degree^2[node])
        deg = len(adj[node])
        delta = defaultdict(int)
        count = defaultdict(int)
        for neigh in adj[node]:
            if len(adj[neigh]) > 0:
                delta[neigh] += \
                    ((deg - 1) * (len(adj[neigh]) - 1)
                     + ci[neigh] // (len(adj[neigh])))
            else:
                # Nodes of degree 1 decrease all the way to zero
                delta[neigh] += ci[neigh]

            for neigh2 in adj[neigh]:
                if neigh2 in adj[node]:
                    delta[neigh2] += len(adj[neigh2])
                else:
                    delta[neigh2] += len(adj[neigh2]) - 1
                count[neigh2] += 1

        # Apply the changes at the same time. Takes O(degree^2[node]).
        for neigh2 in delta:
            # At the end of the previous loop, nodes of type 1 and 2
            # already have the correct deltas, while nodes of type 3 are
            # missing a term.
            if neigh2 in adj[node] and count[neigh2] > 0:
                delta[neigh2] -= count[neigh2]

            # If dict, takes O(1). If heap, takes O(log n)
            ci[neigh2] = ci[neigh2] - delta[neigh2]

        # Finally, finish udpating the graph, and store the node.
        del adj[node]
        removed.append(node)

    return removed


def _immunize_netshield(graph, k):
    """Return the nodes that have the largest 'shield value'.

    References
    ----------

    [1] Chen Chen, Hanghang Tong, B. Aditya Prakash, Charalampos
    E. Tsourakakis, Tina Eliassi-Rad, Christos Faloutsos, Duen Horng Chau:
    Node Immunization on Large Graphs: Theory and Algorithms. IEEE
    Trans. Knowl. Data Eng. 28(1): 113-126 (2016)

    Notes
    -----

    This version only works on graphs with no self-loops.

    """
    # Matrix computations require node labels to be consecutive integers,
    # so we need to (i) convert them if they are not, and (ii) preserve the
    # original labels as an attribute.
    graph = nx.convert_node_labels_to_integers(
        graph, label_attribute='original_label')

    # The variable names here match the notation used in [1].
    A = nx.to_scipy_sparse_matrix(graph).astype('d')
    nodes = [n for n in graph]
    lambda_, u = sparse.linalg.eigs(A, k=1)
    lambda_ = lambda_.real
    u = abs(u.real)
    v = 2 * lambda_ * u**2

    # The first node is just the one with highest eigenvector centrailty
    first_idx = np.argmax(v)
    S = [nodes[first_idx]]
    score = np.zeros(graph.order())
    score[first_idx] = -1
    for _ in range(k - 1):
        # Update score
        B = A[:, S]
        b = B.dot(u[S])
        for idx, j in enumerate(nodes):
            if j in S:
                score[idx] = -1
            else:
                score[idx] = (v[idx] - 2 * b[idx] * u[idx])[0]
        # The remaining nodes are the ones that maximize the score
        S.append(nodes[np.argmax(score)])

    # Return the original labels
    return [graph.nodes[s]['original_label'] for s in S]
