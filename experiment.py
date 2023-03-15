from time import time
import msprime
import tskit
import numpy as np


def timer(func):
    def wrapper(ts):
        t1 = time()
        rv = func(ts)
        t2 = time()
        print(f"Function {func.__name__!r} took {t2 - t1}")
        return rv
    return wrapper


@timer
def times_from_trees(ts: tskit.TreeSequence):
    rv = np.zeros(ts.num_trees)
    for (i, t) in enumerate(ts.trees()):
        rv[i] = t.total_branch_length*t.span

    return rv


@timer
def sample_counts_from_trees(ts: tskit.TreeSequence):
    rv = []
    for (i, t) in enumerate(ts.trees()):
        counts = np.zeros(ts.num_nodes, dtype=np.uint32)
        for node in t.nodes():
            counts[node] = len([i for i in t.leaves(node)])

        rv.append(counts)

    return rv


@timer
def times_from_edge_differences(ts: tskit.TreeSequence):
    rv = np.zeros(ts.num_trees)
    branch_lengths = np.zeros(ts.num_nodes)

    for (i, e) in enumerate(ts.edge_diffs()):
        for edge_out in e.edges_out:
            if edge_out.parent != tskit.NULL:
                branch_lengths[edge_out.child] = 0.0
        for edge_in in e.edges_in:
            if edge_in.parent != tskit.NULL:
                ptime = ts.node(edge_in.parent).time
                ctime = ts.node(edge_in.child).time
                branch_lengths[edge_in.child] = ptime - ctime

        rv[i] = np.sum(branch_lengths)*(e.interval.right - e.interval.left)

    return rv


@timer
def sample_counts_from_edge_differences(ts: tskit.TreeSequence):
    rv = []
    counts = np.zeros(ts.num_nodes, dtype=np.int32)
    for s in ts.samples():
        counts[s] = 1
    parents = [tskit.NULL for i in range(ts.num_nodes)]
    for tree_index, e in enumerate(ts.edge_diffs()):
        for o in e.edges_out:
            assert counts[o.child] <= counts[
                o.parent], f"{o.child} {counts[o.child]} {counts[o.parent]}, {counts}"
            lc = counts[o.child]
            p = parents[o.child]
            while p != tskit.NULL:
                counts[p] -= lc
                p = parents[p]
            parents[o.child] = tskit.NULL

        for edge_in in e.edges_in:
            lc = counts[edge_in.child]
            parents[edge_in.child] = edge_in.parent
            p = parents[edge_in.child]
            while p != tskit.NULL:
                counts[p] += lc
                p = parents[p]

        rv.append(np.copy(counts))

    return rv


ts = msprime.sim_ancestry(1000, recombination_rate=1e-7,
                          sequence_length=1e6, random_seed=666*42, population_size=1000)

assert ts.num_trees > 1

# print(ts.draw_text())

print(f"num trees = {ts.num_trees}")

times_trees = times_from_trees(ts)

times_edge_diffs = times_from_edge_differences(ts)

counts_from_trees = sample_counts_from_trees(ts)

counts_from_edges = sample_counts_from_edge_differences(ts)

assert len(counts_from_trees) == len(counts_from_edges)

for i, j in zip(counts_from_trees, counts_from_edges):
    assert np.all(i - j == 0), f"{np.unique(i - j, return_counts=True)}"
