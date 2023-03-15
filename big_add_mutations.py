from dataclasses import dataclass

import numpy as np

import fwdpy11
import tskit

NMUTATIONS = 250000
# NMUTATIONS = 25000

ts = tskit.load("big.trees")

# NOTE: we cannot globally collect our
# info for all nodes over the tree seq
# as in the vignete. If we try that,
# we will run out of memory.

ttimes = np.zeros(ts.num_trees)

# In experiment.py, we show that this is the
# fastest way to get our times
for i, t in enumerate(ts.trees(sample_lists=False)):
    ttimes[i] = t.total_branch_length*t.span

# normalize
ttimes = ttimes/np.sum(ttimes)

nmutations_per_tree = np.random.multinomial(NMUTATIONS, ttimes, 1)[0]

# Add mutations

desired_VG = 0.8

counts = np.zeros(ts.num_nodes, dtype=np.uint32)
for s in ts.samples():
    counts[s] = 1
parents = [tskit.NULL for i in range(ts.num_nodes)]
branch_lengths = np.zeros(ts.num_nodes)

tables = ts.tables
tables.mutations.metadata_schema = fwdpy11.tskit_tools.metadata_schema.MutationMetadata

positions = set()

check_our_VG = 0.0
check_our_prop_VG = 0.0

for tree_index, e in enumerate(ts.edge_diffs()):
    for o in e.edges_out:
        assert counts[o.child] <= counts[
            o.parent], f"{o.child} {counts[o.child]} {counts[o.parent]}, {counts}"
        lc = counts[o.child]
        p = parents[o.child]
        if p != tskit.NULL:
            branch_lengths[o.child] = 0.0
        while p != tskit.NULL:
            counts[p] -= lc
            p = parents[p]
        parents[o.child] = tskit.NULL

    for edge_in in e.edges_in:
        lc = counts[edge_in.child]
        parents[edge_in.child] = edge_in.parent
        p = parents[edge_in.child]
        if p != tskit.NULL:
            ptime = ts.node(p).time
            ctime = ts.node(edge_in.child).time
            branch_lengths[edge_in.child] = ptime-ctime
        while p != tskit.NULL:
            counts[p] += lc
            p = parents[p]

    if nmutations_per_tree[tree_index] > 0:
        # Mutation frequencies
        # NOTE: this can be done more efficiently
        # by treating it as a branch stat like counts
        freqs = counts/ts.num_samples
        freqs[np.where(freqs == 1.0)] = 0.0
        w = branch_lengths/branch_lengths.sum()

        prop_vg_due_to_tree = nmutations_per_tree[tree_index]/NMUTATIONS

        vg_from_tree = prop_vg_due_to_tree*desired_VG

        check_our_prop_VG += vg_from_tree

        mutations_per_node = np.random.multinomial(
            nmutations_per_tree[tree_index], w, None)
        node_indexes = np.where(mutations_per_node > 0)[0]

        # NOTE: the problem is probably here.
        # Probably. Problem
        vg_per_mutation = []
        for n in node_indexes:
            assert freqs[n] > 0.0
            assert freqs[n] < 1.0
            vg_per_mutation.append(2.0*freqs[n]*(1.0-freqs[n]))
        vg_per_mutation = np.array(vg_per_mutation)
        vg_per_mutation /= vg_per_mutation.sum()
        vg_per_mutation *= vg_from_tree

        # print(vg_per_mutation, mutations_per_node[n])

        for n, vg in zip(node_indexes, vg_per_mutation):
            check_our_VG += vg
            for _ in range(mutations_per_node[n]):
                p = freqs[n]
                esize = np.sqrt((vg/mutations_per_node[n])/(2*p*(1-p)))
                # esize = np.sqrt(vg/(2*p*(1-p)))
                if np.random.uniform(0., 1., 1)[0] < 0.5:
                    esize = -1.0*esize

                pos = np.random.uniform(
                    e.interval.left, e.interval.right, 1)[0]
                while pos in positions:
                    pos = np.random.uniform(
                        e.interval.left, e.interval.right, 1)[0]
                positions.add(pos)
                site = tables.sites.add_row(pos, '1')
                time = int(np.ceil(tables.nodes.time[n]))
                md = {'s': esize,
                      'h': 1.0,
                      'origin': time,
                      'label': np.uint16(0),
                      # NOTE: always write the
                      # next 2 lines as shown here.
                      # The fwdpy11 back end will do
                      # the right thing.
                      # A future release will provide a
                      # nicer API so that you only need
                      # to provide the previous 2 fields.
                      'neutral': 0,
                      'key': np.uint64(0)
                      }
                tables.mutations.add_row(site, n, '1', time=time, metadata=md)

tables.sort()

ts_with_muts = tables.tree_sequence()

ts_with_muts.dump("big_with_muts.trees")

pop = fwdpy11.DiploidPopulation.create_from_tskit(
    ts_with_muts, import_mutations=True)
gvalues = []
for i in pop.diploids:
    g = 0.0
    for k in pop.haploid_genomes[i.first].smutations:
        g += pop.mutations[k].s
    for k in pop.haploid_genomes[i.second].smutations:
        g += pop.mutations[k].s
    gvalues.append(g)

gvalues = np.array(gvalues)

# print(len([i for i in pop.mutations if i.s < 0]),
#       len([i for i in pop.mutations if i.s > 0]))
# print(
#    f"Mean phenotype = {gvalues.mean()},\ninitial VG = {gvalues.var()}.\nDesired initial VG = {desired_VG}")
# print(check_our_VG, desired_VG, check_our_prop_VG)
# print(len(pop.tables.mutations))
