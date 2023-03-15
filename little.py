import msprime

ts = msprime.sim_ancestry(100, recombination_rate=1e-8,
                          sequence_length=3e6, population_size=10000,
                          model=[msprime.DiscreteTimeWrightFisher(duration=100),
                                 msprime.StandardCoalescent()])


ts.dump("little.trees")
