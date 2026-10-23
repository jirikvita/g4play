// Water Cherenkov tank: 4 mm stainless steel cylinder (D=3.5 m, H=4 m) filled with water.
// Usage: ./waterCherenkovSim [macro.mac]
// Switch particle with /gun/particle e-|e+|mu-|mu+|pi-|pi+|proton|anti_proton and /gun/energy.

#include "G4RunManagerFactory.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VUserActionInitialization.hh"
#include "FTFP_BERT.hh"
#include "G4OpticalPhysics.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4VisAttributes.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "G4UserRunAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4AnalysisManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4VProcess.hh"

#include <cstdlib>
#include <vector>

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* water = nist->FindOrBuildMaterial("G4_WATER");
    auto* steel = nist->FindOrBuildMaterial("G4_STAINLESS-STEEL");

    // Refractive index of water (needed for Cherenkov emission)
    std::vector<G4double> energy = {1.9 * eV, 2.3 * eV, 2.8 * eV, 3.3 * eV, 4.0 * eV};
    std::vector<G4double> rindex = {1.331, 1.337, 1.345, 1.354, 1.370};
    std::vector<G4double> abslen = {30. * m, 30. * m, 20. * m, 10. * m, 5. * m};
    auto* wmpt = new G4MaterialPropertiesTable();
    wmpt->AddProperty("RINDEX", energy, rindex);
    wmpt->AddProperty("ABSLENGTH", energy, abslen);
    water->SetMaterialPropertiesTable(wmpt);

    std::vector<G4double> airIndex(energy.size(), 1.0);
    auto* ampt = new G4MaterialPropertiesTable();
    ampt->AddProperty("RINDEX", energy, airIndex);
    air->SetMaterialPropertiesTable(ampt);

    const G4double outerR = 1.75 * m;
    const G4double halfH = 2.0 * m;
    const G4double wall = 4. * mm;

    auto* worldS = new G4Box("World", 3. * m, 3. * m, 4. * m);
    auto* worldLV = new G4LogicalVolume(worldS, air, "World");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0, true);

    // Steel shell: full outer cylinder, water placed inside leaves 4 mm walls and caps
    auto* tankS = new G4Tubs("Tank", 0., outerR, halfH, 0., 360. * deg);
    auto* tankLV = new G4LogicalVolume(tankS, steel, "Tank");
    new G4PVPlacement(nullptr, {}, tankLV, "Tank", worldLV, false, 0, true);

    auto* waterS = new G4Tubs("Water", 0., outerR - wall, halfH - wall, 0., 360. * deg);
    auto* waterLV = new G4LogicalVolume(waterS, water, "Water");
    new G4PVPlacement(nullptr, {}, waterLV, "Water", tankLV, false, 0, true);

    tankLV->SetVisAttributes(new G4VisAttributes(G4Colour(0.6, 0.6, 0.6)));
    auto* wva = new G4VisAttributes(G4Colour(0., 0.3, 1., 0.3));
    wva->SetForceSolid(true);
    waterLV->SetVisAttributes(wva);

    return worldPV;
  }
};

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
public:
  PrimaryGenerator() : fGun(new G4ParticleGun(1)) {
    fGun->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("mu-"));
    fGun->SetParticleEnergy(1. * GeV);
    // 10 cm outside the tank wall at x = -R, mid-height, aimed at the tank centre
    fGun->SetParticleMomentumDirection(G4ThreeVector(1., 0., 0.));
    fGun->SetParticlePosition(G4ThreeVector(-(1.75 * m + 10. * cm), 0., 0.));
  }
  ~PrimaryGenerator() override { delete fGun; }
  void GeneratePrimaries(G4Event* event) override { fGun->GeneratePrimaryVertex(event); }

private:
  G4ParticleGun* fGun;
};

// Opens muonDecay.root with an ntuple "muonDecay"; worker ntuples are merged into one file.
class RunAction : public G4UserRunAction {
public:
  RunAction() {
    auto* am = G4AnalysisManager::Instance();
    am->SetNtupleMerging(true);
    am->SetVerboseLevel(0);
    am->CreateNtuple("muonDecay", "Last muon point before decay");
    am->CreateNtupleIColumn("eventID");
    am->CreateNtupleIColumn("trackID");
    am->CreateNtupleIColumn("pdg");
    am->CreateNtupleDColumn("x_mm");
    am->CreateNtupleDColumn("y_mm");
    am->CreateNtupleDColumn("z_mm");
    am->CreateNtupleDColumn("t_ns");
    am->CreateNtupleDColumn("ekin_MeV");
    am->FinishNtuple();
  }
  void BeginOfRunAction(const G4Run*) override {
    const char* out = std::getenv("MUON_OUT");
    G4AnalysisManager::Instance()->OpenFile(out ? out : "muonDecay.root");
  }
  void EndOfRunAction(const G4Run*) override {
    auto* am = G4AnalysisManager::Instance();
    am->Write();
    am->CloseFile();
  }
};

// Fills one row per muon at the step in which it decays (post-step point = last track point).
class SteppingAction : public G4UserSteppingAction {
public:
  void UserSteppingAction(const G4Step* step) override {
    const auto* track = step->GetTrack();
    if (std::abs(track->GetDefinition()->GetPDGEncoding()) != 13) return;
    const auto* post = step->GetPostStepPoint();
    const auto* proc = post->GetProcessDefinedStep();
    // Stopped mu- end via capture-at-rest in Geant4 (it never gets a "Decay" step).
    if (!proc || (proc->GetProcessName() != "Decay" && proc->GetProcessName() != "muMinusCaptureAtRest")) return;

    const auto pos = post->GetPosition();
    auto* am = G4AnalysisManager::Instance();
    am->FillNtupleIColumn(0, G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
    am->FillNtupleIColumn(1, track->GetTrackID());
    am->FillNtupleIColumn(2, track->GetDefinition()->GetPDGEncoding());
    am->FillNtupleDColumn(3, pos.x() / mm);
    am->FillNtupleDColumn(4, pos.y() / mm);
    am->FillNtupleDColumn(5, pos.z() / mm);
    am->FillNtupleDColumn(6, post->GetGlobalTime() / ns);
    am->FillNtupleDColumn(7, post->GetKineticEnergy() / MeV);
    am->AddNtupleRow();
  }
};

class ActionInitialization : public G4VUserActionInitialization {
public:
  void BuildForMaster() const override { SetUserAction(new RunAction()); }
  void Build() const override {
    SetUserAction(new PrimaryGenerator());
    SetUserAction(new RunAction());
    SetUserAction(new SteppingAction());
  }
};

// Usage: ./waterCherenkovSim [macro.mac] [-i]   (-i keeps the viewer open after the macro)
int main(int argc, char** argv) {
  G4String macro;
  bool interactive = (argc == 1);
  for (int i = 1; i < argc; ++i) {
    if (G4String(argv[i]) == "-i") interactive = true;
    else macro = argv[i];
  }
  // This build may lack OpenGL/Qt; VRML2FILE is the fallback. Macros read it via /control/getEnv.
  setenv("VIS_DRIVER", "VRML2FILE", 0);
  G4UIExecutive* ui = interactive ? new G4UIExecutive(argc, argv) : nullptr;

  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new DetectorConstruction());

  auto* physics = new FTFP_BERT();
  physics->RegisterPhysics(new G4OpticalPhysics());
  runManager->SetUserInitialization(physics);
  runManager->SetUserInitialization(new ActionInitialization());

  auto* visManager = new G4VisExecutive();
  visManager->Initialize();

  auto* uiManager = G4UImanager::GetUIpointer();
  if (!macro.empty()) {
    uiManager->ApplyCommand("/control/execute " + macro);
    if (ui) {
      ui->SessionStart();
      delete ui;
    }
  } else {
    uiManager->ApplyCommand("/run/initialize");
    uiManager->ApplyCommand("/control/getEnv VIS_DRIVER");
    uiManager->ApplyCommand("/vis/open {VIS_DRIVER}");
    uiManager->ApplyCommand("/vis/drawVolume");
    uiManager->ApplyCommand("/vis/viewer/set/viewpointThetaPhi 70 20");
    uiManager->ApplyCommand("/vis/scene/add/trajectories smooth");
    uiManager->ApplyCommand("/vis/scene/endOfEventAction accumulate");
    ui->SessionStart();
    delete ui;
  }

  delete visManager;
  delete runManager;
  return 0;
}
