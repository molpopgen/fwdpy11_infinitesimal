import msprime
import tskit
import numpy as np

seed = 12345

NSAMPLES = 500
NTREES = 100000

treeseqs = []

for ts in msprime.sim_ancestry(NSAMPLES, sequence_length=1e8,
                               recombination_rate=0.0, random_seed=seed, num_replicates=NTREES):
    treeseqs.append(ts)
# ts = msprime.sim_ancestry(2500, sequence_length=1e8,
#                          recombination_rate=5e-5)
# print(ts.num_trees)

lens = []

for ts in treeseqs:
    for t in ts.trees():
        lens.append(t.total_branch_length * t.span/ts.sequence_length)

lens = np.array(lens)
lens = lens / np.sum(lens)

VG = 1.0

vgs = []

rvg = VG
for i, l in enumerate(lens):
    remaining = np.sum(lens[i:])
    #p = l/remaining
    #vg = np.random.normal(p*rvg, p*(1.-p)*rvg/2)
    #rvg -= vg
    vg = l*VG
    vgs.append(vg)

print(rvg, np.sum(vgs))

esizes = []
counts = []
infinite_sites = set()
pheno = [0.]*ts.num_individuals
for v, ts in zip(vgs, treeseqs):
    time = ts.nodes_time
    for t in ts.trees(sample_lists=True):
        nodes = []
        blens = []
        p = t.parent_array
        for node in t.nodes():
            pnode = p[node]
            if pnode != tskit.NULL:
                nodes.append(node)
                blens.append(time[pnode] - time[node])

        blens = np.array(blens)
        w = blens/blens.sum()
        branch = np.random.choice(nodes, 1, p=w)[0]
        n = len([i for i in t.samples(branch)])
        q = n / ts.num_samples

        # get effect size from the variance
        esize = np.sqrt(v/(2*q*(1.-q)))
        genotypes = np.zeros(ts.num_individuals)
        for n in t.samples(branch):
            genotypes[ts.node(n).individual] += 1
        p = 1.-q
        counts = np.unique(genotypes, return_counts=True)
        denom = 0.0
        for i, j in zip(counts[0], counts[1]):
            if i == 1:
                denom += j/ts.num_individuals
            elif i == 2:
                denom += 4*j/ts.num_individuals
        esize2 = np.sqrt(v/denom)
        #print(esize, esize2)
        esize = esize2
        # esize = np.sqrt(v/(2.*q*(1.-q + 2.*q)))
        # print(esize, esize1, esize - esize1)
        # :w = np.sqrt(v/(2.*q))
        # e3 = np.sqrt(v/(q*(1.-q)))
        # print(esize, e2, e3)

        #x = np.array([0]*(ts.num_samples - n) + [esize]*n)

        # symmetry...
        if np.random.uniform(0, 1, 1)[0] < 0.5:
            esize = -esize

        for s in t.samples(branch):
            tsnode = ts.node(s)
            tsnode.time == 0.0
            ind = tsnode.individual
            pheno[ind] += esize

        # print(esize, q, v)  # , x.var())
        # pos = np.random.uniform(t.interval.left, t.interval.right, 1)[0]
        # while pos in infinite_sites:
        #     pos = np.random.uniform(t.interval.left, t.interval.right, 1)[0]
        # infinite_sites.add(pos)
        # site = tcopy.sites.add_row(pos, '0')
        # tcopy.mutations.add_row(site, branch, '1')
        # esizes.append(esize)
        # counts.append(n)

pheno = np.array(pheno)
print(len(pheno), ts.num_samples)
print(pheno.mean(), pheno.var())

# tcopy.sort()
# ts2 = tcopy.tree_sequence()
#
# for t in ts2.trees(sample_lists=True):
#     assert t.num_mutations == 1
#     for m in t.mutations():
#         c = 0
#         for s in t.samples(m.node):
#             p[s] += esizes[m.id]
#             c += 1
#         assert c == counts[m.id], f"{c} {counts[m.id]} {counts} {t.index}"
#
# p = np.array(p)
# print(p.mean(), p.var())
# print(len([i for i in esizes if i > 0.0]), len(esizes))
#
# for i in p:
#     print('p', i)
# for i, j in zip(counts, esizes):
#     print('m', i, j)
