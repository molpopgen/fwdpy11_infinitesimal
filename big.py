import msprime

ts = msprime.sim_ancestry(10000, recombination_rate=1e-8,
                          sequence_length=3e9, population_size=10000,
                          model=[msprime.DiscreteTimeWrightFisher(duration=1000),
                                 msprime.StandardCoalescent()])


ts.dump("big.trees")
