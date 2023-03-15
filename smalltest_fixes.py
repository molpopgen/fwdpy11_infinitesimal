import msprime
import tskit
import numpy as np

seed = 12345
ts = msprime.sim_ancestry(5000, sequence_length=1e8,
                          recombination_rate=5e-8, random_seed=seed)
# ts = msprime.sim_ancestry(5000, sequence_length=1e8,
#                           recombination_rate=1e-5)
print(ts.num_trees)

lens = []

for t in ts.trees():
    lens.append(t.total_branch_length * t.span/ts.sequence_length)

lens = np.array(lens)
lens = lens / np.sum(lens)

VG = 1.0

vgs = []

rvg = VG
for i, l in enumerate(lens):
    # remaining = np.sum(lens[i:])
    # p = l/remaining
    # vg = np.random.normal(p*rvg, p*(1.-p)*rvg/2)
    vg = l*VG
    assert vg >= 0.0
    # rvg -= vg
    vgs.append(vg)

print(rvg, np.sum(vgs))

time = ts.nodes_time
tcopy = ts.tables
esizes = []
counts = []
infinite_sites = set()
for v, tree in zip(vgs, ts.trees(sample_lists=True)):
    nodes = []
    blens = []
    p = tree.parent_array
    for node in tree.nodes():
        pnode = p[node]
        if pnode != tskit.NULL:
            nodes.append(node)
            blens.append(time[pnode] - time[node])
            assert blens[-1] > 0.0

    blens = np.array(blens)
    w = blens/blens.sum()
    branch = np.random.choice(nodes, 1, p=w)[0]
    n = len([i for i in tree.samples(branch)])
    q = n / ts.num_samples

    #esize = np.sqrt(v/(q*(1.-q)))  # get effect size from the variance
    #esize = np.sqrt(v/q)
    p = 1.-q
    esize = np.sqrt(v/(1.*p*q + 2.*q*q))

    # x = np.array([0]*(ts.num_samples - n) + [esize]*n)

    # symmetry...
    if np.random.uniform(0, 1, 1)[0] < 0.5:
        esize = -esize
    # print(esize, q, v)  # , x.var())
    pos = np.random.uniform(tree.interval.left, tree.interval.right, 1)[0]
    while pos in infinite_sites:
        pos = np.random.uniform(tree.interval.left, tree.interval.right, 1)[0]
    infinite_sites.add(pos)
    site = tcopy.sites.add_row(pos, '0')
    tcopy.mutations.add_row(site, branch, '1')
    esizes.append(esize)
    counts.append(n)

tcopy.sort()
ts2 = tcopy.tree_sequence()

p = [0.]*ts2.num_samples
nmuts = [0]*ts2.num_samples
last_mut_id = -1
for t in ts2.trees(sample_lists=True):
    assert t.num_mutations == 1
    for m in t.mutations():
        assert m.id == last_mut_id + 1
        last_mut_id += 1
        c = 0
        for s in t.samples(m.node):
            p[s] += esizes[m.id]
            nmuts[s] += 1
            c += 1
        assert c == counts[m.id], f"{c} {counts[m.id]} {counts} {t.index}"

p = np.array(p)
print(p.mean(), p.var())
print(len([i for i in esizes if i > 0.0]), len(esizes))

for i, j in zip(p, nmuts):
    print('p', i, j)
for i, j in zip(counts, esizes):
    print('m', i, j)
