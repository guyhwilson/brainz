from __future__ import division, print_function
import numpy as np

def modularity_louvain_und_sign(W, gamma=1, qtype='sta', seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    The Louvain algorithm is a fast and accurate community detection
    algorithm (at the time of writing).

    Use this function as opposed to modularity_louvain_und() only if the
    network contains a mix of positive and negative weights.  If the network
    contains all positive weights, the output will be equivalent to that of
    modularity_louvain_und().

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # weight of positive links
    s1 = np.sum(W1)  # weight of negative links

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-sQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = d0  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    h = 1  # hierarchy index
    nh = n  # number of nodes in hierarchy
    ci = [None, np.arange(n) + 1]  # hierarchical module assignments
    q = [-1, 0]  # hierarchical modularity values
    while q[h] - q[h - 1] > 1e-10:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style A.  Please '
                                'contact the developer with this error.')
        kn0 = np.sum(W0, axis=0)  # positive node degree
        kn1 = np.sum(W1, axis=0)  # negative node degree
        km0 = kn0.copy()  # positive module degree
        km1 = kn1.copy()  # negative module degree
        knm0 = W0.copy()  # positive node-to-module degree
        knm1 = W1.copy()  # negative node-to-module degree

        m = np.arange(nh) + 1  # initial module assignments
        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Infinite Loop was detected and stopped. '
                                    'This was probably caused by passing in a directed matrix.')
            flag = False
            # loop over nodes in random order
            for u in np.random.permutation(nh):
                ma = m[u] - 1
                dQ0 = ((knm0[u, :] + W0[u, u] - knm0[u, ma]) -
                       gamma * kn0[u] * (km0 + kn0[u] - km0[ma]) / s0)  # positive dQ
                dQ1 = ((knm1[u, :] + W1[u, u] - knm1[u, ma]) -
                       gamma * kn1[u] * (km1 + kn1[u] - km1[ma]) / s1)  # negative dQ

                dQ = d0 * dQ0 - d1 * dQ1  # rescaled changes in modularity
                dQ[ma] = 0  # no changes for same module

                max_dQ = np.max(dQ)  # maximal increase in modularity
                if max_dQ > 1e-10:  # if maximal increase is positive
                    flag = True
                    mb = np.argmax(dQ)

                    # change positive node-to-module degrees
                    knm0[:, mb] += W0[:, u]
                    knm0[:, ma] -= W0[:, u]
                    # change negative node-to-module degrees
                    knm1[:, mb] += W1[:, u]
                    knm1[:, ma] -= W1[:, u]
                    km0[mb] += kn0[u]  # change positive module degrees
                    km0[ma] -= kn0[u]
                    km1[mb] += kn1[u]  # change negative module degrees
                    km1[ma] -= kn1[u]

                    m[u] = mb + 1  # reassign module

        h += 1
        ci.append(np.zeros((n,)))
        _, m = np.unique(m, return_inverse=True)
        m += 1

        for u in range(nh):  # loop through initial module assignments
            ci[h][np.where(ci[h - 1] == u + 1)] = m[u]  # assign new modules

        nh = np.max(m)  # number of new nodes
        wn0 = np.zeros((nh, nh))  # new positive weights matrix
        wn1 = np.zeros((nh, nh))

        for u in range(nh):
            for v in range(u, nh):
                wn0[u, v] = np.sum(W0[np.ix_(m == u + 1, m == v + 1)])
                wn1[u, v] = np.sum(W1[np.ix_(m == u + 1, m == v + 1)])
                wn0[v, u] = wn0[u, v]
                wn1[v, u] = wn1[u, v]

        W0 = wn0
        W1 = wn1

        q.append(0)
        # compute modularity
        q0 = np.trace(W0) - np.sum(np.dot(W0, W0)) / s0
        q1 = np.trace(W1) - np.sum(np.dot(W1, W1)) / s1
        q[h] = d0 * q0 - d1 * q1

    _, ci_ret = np.unique(ci[-1], return_inverse=True)
    ci_ret += 1

    return ci_ret, q[-1]