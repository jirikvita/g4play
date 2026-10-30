
Training NN (Python)
====================

Use the project virtual environment in this folder and install the required packages before running [trainNN.py](/home/qitek/work/github/g4play/trainNN.py):

```
cd /home/qitek/work/github/g4play
/home/qitek/work/github/g4play/.venv/bin/pip install -r requirements.txt
/home/qitek/work/github/g4play/.venv/bin/python trainNN.py
```

source /home/qitek/install/geant4-v11.1.1-install/bin/geant4.sh
cd /home/qitek/work/github/g4play
make clean && make
VIS_DRIVER=OGL ./waterCherenkovSim

and in the command line of the GUI:

/gun/particle mu-
/gun/energy 1 GeV
/run/beamOn 1


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
