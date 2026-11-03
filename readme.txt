
Training NN (Python)
====================

Use the project virtual environment in this folder and install the required packages before running [trainNN.py](/home/qitek/work/github/g4play/trainNN.py):

```
cd /home/qitek/work/github/g4play
/home/qitek/work/github/g4play/.venv/bin/pip install -r requirements.txt
/home/qitek/work/github/g4play/.venv/bin/python trainNN.py
```

Batch apply NN on a directory
=============================

Run [nnTestAll.py](/home/qitek/work/g4play/nnTestAll.py) to process all `.root` files except
`*_nnPID.root` in `test_only_1k/` by default:

```
python3 nnTestAll.py
```

Pick another directory with `--dir`:

```
python3 nnTestAll.py --dir train_test
```

For each tested ROOT input, [applyNN.py](/home/qitek/work/g4play/applyNN.py) also writes
`*.momentum_hist.csv`, `*.momentum_hist.png`, and `*.momentum_hist.pdf`. After all files are
processed, [nnTestAll.py](/home/qitek/work/g4play/nnTestAll.py) creates
`all_tests_momentum_hists.pdf` in the selected `--dir`.

[trainNN.py](/home/qitek/work/g4play/trainNN.py) saves models under [models/](/home/qitek/work/g4play/models)
and names the exported model with its hidden-layer sizes, for example
`models/pid_model_h256x128.pt` for `--hidden 256,128`.
Associated outputs use the same base name:
`models/pid_model_h256x128.momentum_overlay.png` and
`models/pid_model_h256x128.momentum_overlay.pdf`
(filled histograms with `alpha=0.3`, with true-momentum labels).

[nnTestAll.py](/home/qitek/work/g4play/nnTestAll.py) also writes:
`all_tests_inferred_momentum_overlay.png` and `all_tests_inferred_momentum_overlay.pdf`
with one filled histogram per tested ROOT file (`alpha=0.3`) and legend labels including the true
momentum from the filename.

Batch testing across standard directories
=========================================

Use [superTest.py](/home/qitek/work/g4play/superTest.py) to run
[nnTestAll.py](/home/qitek/work/g4play/nnTestAll.py) across the predefined test directories:

```
python3 superTest.py --model models/pid_model_h256x128.pt
```

For each tested directory, generated outputs (`*_nnPID*` and `all_tests_*`) are moved into
`model_<model-name>/` (for example `model_pid_model_h256x128/`), and
[CreateHTMLpage_noConv.sh](/home/qitek/work/g4play/scripts/CreateHTMLpage_noConv.sh)
is run there with the HTML title set to the model filename.

Parallel Geant4 scans
=====================

Use [superRun.py](/home/qitek/work/g4play/superRun.py) to run [run_scan.py](/home/qitek/work/g4play/run_scan.py)
for multiple particle types simultaneously. Defaults to `e+`, `mu+`, and `pi+`:

```
python3 superRun.py
```

With custom momenta:

```
python3 superRun.py 200 300 400
```

Dry-run only:

```
python3 superRun.py -n 200
```

source /home/qitek/install/geant4-v11.1.1-install/bin/geant4.sh
cd /home/qitek/work/github/g4play
make clean && make
VIS_DRIVER=OGL ./waterCherenkovSim

and in the command line of the GUI:

/gun/particle mu-
/gun/energy 1 GeV
/g4play/silicon/smearPoissonHits true
/g4play/silicon/addNoise false
/g4play/silicon/noiseNu 0.4
/run/beamOn 1

The silicon readout command `/g4play/silicon/smearPoissonHits` is optional and defaults to `true`.
Set it to `false` to store the original (unsmeared) per-module hit counts.
When Poisson smearing is enabled, output ROOT filenames get the suffix `_PoissSmear`.
Optional `/g4play/silicon/addNoise true` adds Poisson noise per module with mean
`/g4play/silicon/noiseNu` (default `0.4`), and the output filename gets suffix `_addNoise{nu}`.


# exclude photons from view:
/vis/filtering/trajectories/create/particleFilter
/vis/filtering/trajectories/particleFilter-0/add opticalphoton
/vis/filtering/trajectories/particleFilter-0/invert true


# select particles only:
/vis/filtering/trajectories/particleFilter-0/add mu-
/vis/filtering/trajectories/particleFilter-0/add e-



./waterCherenkovSim run.mac now creates muonDecay.root (about 47 KB). The muonDecay tree has 100 entries, one per event, for the 100 mu- at 300 MeV in your run.mac.

The first run didn't create the file because of the mu- in your current run.mac. In Geant4 a stopped mu- ends in muMinusCaptureAtRest and never gets a Decay step. With 100 mu- no rows



No, it doesn't decay in the simulation. What's saved for mu- is where it stops and is captured.

In Geant4 (FTFP_BERT) a mu- that comes to rest in water is ended by the muMinusCaptureAtRest process, not by Decay. In the 20-event mu- trace I checked, every stopped mu- ended that way and none had a Decay step.

The saved row for a mu- is therefore its stopping point, with ekin_MeV = 0. Real bound mu- in water mostly decay in orbit (about 80%) and are captured by the nucleus only about 20% of the time. This simulation applies capture to all of them, so it doesn't reproduce that decay fraction.

My earlier wording "can be captured" understated this, since in these runs all of them were.

For physics that depends on the decay fraction, the pdg column separates the cases. A pdg of −13 is a mu+, which always gets a true Decay step, and 13 is a mu-, whose saved point is the stopping point.

If you want only true decays, revert the muMinusCaptureAtRest condition. mu- would then never be saved. If you want the decay-in-orbit fraction, that needs a different muon-capture physics setup, which I can look into.
