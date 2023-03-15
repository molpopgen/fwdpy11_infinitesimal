from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import demes
import fwdpy11
import tskit

split = """
description: split model
time_units: generations
demes:
  - name: ancestor
    epochs:
      - start_size: 10000
        end_time: 100
  - name: derived1
    ancestors: [ancestor]
    epochs:
      - start_size: 10000
  - name: derived2
    ancestors: [ancestor]
    epochs:
      - start_size: 10000
"""


@dataclass
class Data:
    generation: int
    deme: int
    gbar: float
    vg: float
    wbar: float


@dataclass
class Recorder:
    data: list

    def __call__(self, pop, _):
        md = np.array(pop.diploid_metadata)
        for deme_index in [1, 2]:
            deme = np.where(md["deme"] == deme_index)[0]
            self.data.append(Data(pop.generation, deme_index,
                                  md["g"][deme].mean(),
                                  md["g"][deme].var(),
                                  md["w"][deme].mean()))
            print(pop.generation, self.data[-1])


def load_and_create(filename: str):
    """
    This function is important because
    it ensures that the variable ts
    is released from memory when it goes
    out of scope.
    """
    ts = tskit.load(filename)
    return fwdpy11.DiploidPopulation.create_from_tskit(ts, import_mutations=True)


def work(filename: str):
    pop = load_and_create(filename)

    graph = demes.loads(split)
    # No burn in: that has been approximated using msprime.
    model = fwdpy11.discrete_demography.from_demes(graph, 0)

    pdict = {
        'nregions': [],
        'sregions': [],
        'recregions': [fwdpy11.PoissonInterval(0.0, pop.tables.genome_length, 1e-8*3e9)],
        # parameters of the stabilizing selection model
        # in ancestor, derived1, and derived2, respectively.
        'gvalue': [fwdpy11.Additive(2., fwdpy11.GSS(optimum=0., VS=60.)),
                   fwdpy11.Additive(2., fwdpy11.GSS(optimum=-5., VS=60.)),
                   fwdpy11.Additive(2., fwdpy11.GSS(optimum=5., VS=60.))],
        'rates': (0, 0, None),
        'demography': model,
        'simlen': model.metadata['total_simulation_length'],
        'prune_selected': False
    }

    params = fwdpy11.ModelParams(**pdict)

    rng = fwdpy11.GSLrng(1252)

    recorder = Recorder([])

    fwdpy11.evolvets(rng, pop, params, 50, recorder=recorder)

    ts = pop.dump_tables_to_tskit()

    ts.dump("big_evolved_fwd.trees")

    fig, (gbar, vg, wbar) = plt.subplots(1, 3)

    for deme_id in [1, 2]:
        gbar.plot([i.generation for i in recorder.data if i.deme == deme_id],
                  [i.gbar for i in recorder.data if i.deme == deme_id])
        vg.plot([i.generation for i in recorder.data if i.deme == deme_id],
                [i.vg for i in recorder.data if i.deme == deme_id])
        wbar.plot([i.generation for i in recorder.data if i.deme == deme_id],
                  [i.wbar for i in recorder.data if i.deme == deme_id])

    plt.savefig("plots.png")


if __name__ == "__main__":
    work("big_with_muts.trees")
