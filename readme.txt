
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
