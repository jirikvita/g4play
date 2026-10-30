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
#include "G4OpBoundaryProcess.hh"
#include "G4ProcessManager.hh"
#include "G4Sphere.hh"
#include "G4VSensitiveDetector.hh"
#include "G4SDManager.hh"
#include "G4UserEventAction.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"
#include "G4PhysicalConstants.hh"
#include "G4TouchableHistory.hh"
#include "G4OpticalPhoton.hh"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <vector>

// Gun direction of the current event (per thread), used by the stepping action.
thread_local G4ThreeVector gGunDir(1., 0., 0.);

// Silicon centre of every photon sensor module (index = copy number of the vacuum volume).
std::vector<G4ThreeVector> gSiPos;
// Per-thread optical photon hit counts of the current event: module index -> hits.
struct SiHit { int n = 0; double tFirst = 0., tSum = 0.; };
thread_local std::map<int, SiHit> gSiHits;

// Counts optical/UV photons absorbed at the silicon surface (flagged by the skin surface).
class SiliconSD : public G4VSensitiveDetector {
public:
  explicit SiliconSD(const G4String& name) : G4VSensitiveDetector(name) {}
  G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
    if (step->GetTrack()->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition()) return false;
    // Depth 0 = silicon, depth 1 = vacuum volume carrying the module index.
    const auto* touch = step->GetPostStepPoint()->GetTouchable();
    // Global time is measured from the primary generation (gun time = 0)
    const double t = step->GetPostStepPoint()->GetGlobalTime();
    auto& h = gSiHits[touch->GetCopyNumber(1)];
    h.tFirst = h.n == 0 ? t : std::min(h.tFirst, t);
    h.tSum += t;
    ++h.n;
    return true;
  }
};

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

    BuildSensors(waterLV, outerR - wall, halfH - wall);

    return worldPV;
  }

  void ConstructSDandField() override {
    auto* sd = new SiliconSD("SiliconSD");
    G4SDManager::GetSDMpointer()->AddNewDetector(sd);
    SetSensitiveDetector("SiliconLV", sd);
  }

private:
  // Half-spherical glass modules (D = 8 cm, 1 mm shell, vacuum inside, 5 mm silicon base)
  // on a 10 cm grid over the inner barrel wall and both end caps, domes pointing inward.
  void BuildSensors(G4LogicalVolume* mother, G4double rIn, G4double halfZ) {
    auto* nist = G4NistManager::Instance();
    auto* glass = nist->FindOrBuildMaterial("G4_Pyrex_Glass");
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* silicon = nist->FindOrBuildMaterial("G4_Si");

    std::vector<G4double> e = {1.9 * eV, 2.3 * eV, 2.8 * eV, 3.3 * eV, 4.0 * eV};
    auto* gmpt = new G4MaterialPropertiesTable();
    gmpt->AddProperty("RINDEX", e, std::vector<G4double>(e.size(), 1.5));
    gmpt->AddProperty("ABSLENGTH", e, std::vector<G4double>(e.size(), 1. * m));
    glass->SetMaterialPropertiesTable(gmpt);
    auto* vmpt = new G4MaterialPropertiesTable();
    vmpt->AddProperty("RINDEX", e, std::vector<G4double>(e.size(), 1.0));
    vacuum->SetMaterialPropertiesTable(vmpt);

    const G4double rOut = 4. * cm, thick = 1. * mm, siT = 5. * mm, rSi = 3.8 * cm;
    const G4double rInner = rOut - thick;
    const G4double gap = 0.5 * mm;  // keeps the flat base off the curved steel wall
    const G4double pitch = 10. * cm;

    auto* glassS = new G4Sphere("Glass", rInner, rOut, 0., twopi, 0., halfpi);
    auto* vacS = new G4Sphere("Vacuum", 0., rInner, 0., twopi, 0., halfpi);
    auto* siS = new G4Tubs("Silicon", 0., rSi, siT / 2, 0., twopi);
    auto* glassLV = new G4LogicalVolume(glassS, glass, "GlassLV");
    auto* vacLV = new G4LogicalVolume(vacS, vacuum, "VacuumLV");
    auto* siLV = new G4LogicalVolume(siS, silicon, "SiliconLV");
    new G4PVPlacement(nullptr, G4ThreeVector(0., 0., siT / 2), siLV, "Silicon", vacLV, false, 0, false);

    // Perfect absorber that registers a detection for every photon reaching the silicon
    auto* surf = new G4OpticalSurface("SiSurface", unified, polished, dielectric_metal);
    auto* smpt = new G4MaterialPropertiesTable();
    smpt->AddProperty("REFLECTIVITY", e, std::vector<G4double>(e.size(), 0.));
    smpt->AddProperty("EFFICIENCY", e, std::vector<G4double>(e.size(), 1.));
    surf->SetMaterialPropertiesTable(smpt);
    new G4LogicalSkinSurface("SiSkin", siLV, surf);

    auto* gva = new G4VisAttributes(G4Colour(0.8, 0.9, 1., 0.4));
    glassLV->SetVisAttributes(gva);
    vacLV->SetVisAttributes(G4VisAttributes::GetInvisible());
    siLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.2, 0.2)));

    int copy = 0;
    auto place = [&](const G4ThreeVector& wallPoint, const G4ThreeVector& inward) {
      G4ThreeVector xa = inward.orthogonal().unit();
      G4RotationMatrix rot(xa, inward.cross(xa), inward);
      const G4ThreeVector origin = wallPoint + gap * inward;
      G4Transform3D tr(rot, origin);
      new G4PVPlacement(tr, glassLV, "Glass", mother, false, copy, false);
      new G4PVPlacement(tr, vacLV, "Vacuum", mother, false, copy, false);
      gSiPos.push_back(origin + (siT / 2) * inward);
      ++copy;
    };

    // Barrel: ~10 cm arc pitch around, 10 cm in z, kept 10 cm clear of the caps
    const int nPhi = int(twopi * rIn / pitch);
    const int nz = int((halfZ - 10. * cm) / pitch);
    for (int k = -nz; k <= nz; ++k) {
      for (int i = 0; i < nPhi; ++i) {
        const G4double phi = twopi * i / nPhi;
        const G4ThreeVector radial(std::cos(phi), std::sin(phi), 0.);
        place(G4ThreeVector(rIn * radial.x(), rIn * radial.y(), k * pitch), -radial);
      }
    }
    // End caps
    const int nxy = int((rIn - rOut) / pitch);
    for (int sgn : {-1, 1}) {
      for (int i = -nxy - 1; i <= nxy + 1; ++i) {
        for (int j = -nxy - 1; j <= nxy + 1; ++j) {
          const G4double x = i * pitch, y = j * pitch;
          if (std::hypot(x, y) + rOut > rIn) continue;
          place(G4ThreeVector(x, y, sgn * halfZ), G4ThreeVector(0., 0., -sgn));
        }
      }
    }
    G4cout << "Sensor modules placed: " << copy << G4endl;
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
  void GeneratePrimaries(G4Event* event) override {
    gGunDir = fGun->GetParticleMomentumDirection();
    fGun->GeneratePrimaryVertex(event);
  }

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
    am->CreateNtupleDColumn("angle_deg");  // -1 if the muon never crossed the plane
    am->FinishNtuple();

    // One row per event; vector columns list the silicon centres with >= 1 photon hit.
    am->CreateNtuple("silicon", "Silicon sensors with photon hits");
    am->CreateNtupleIColumn(1, "eventID");
    am->CreateNtupleDColumn(1, "x_mm", gX);
    am->CreateNtupleDColumn(1, "y_mm", gY);
    am->CreateNtupleDColumn(1, "z_mm", gZ);
    am->CreateNtupleIColumn(1, "nHits", gN);
    am->CreateNtupleDColumn(1, "tFirst_ns", gTFirst);  // earliest photon arrival
    am->CreateNtupleDColumn(1, "tMean_ns", gTMean);    // mean photon arrival time
    am->CreateNtupleIColumn(1, "primaryPDG");          // PDG code of the generated particle
    am->FinishNtuple(1);
  }
  static inline thread_local std::vector<double> gX, gY, gZ, gTFirst, gTMean;
  static inline thread_local std::vector<int> gN;
  void BeginOfRunAction(const G4Run*) override {
    // Let detected photons call the silicon sensitive detector (off by default)
    auto* pv = G4OpticalPhoton::OpticalPhotonDefinition()->GetProcessManager()->GetProcessList();
    for (std::size_t i = 0; i < pv->size(); ++i) {
      if (auto* bp = dynamic_cast<G4OpBoundaryProcess*>((*pv)[i])) bp->SetInvokeSD(true);
    }
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

    const int evt = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    if (evt != fEvent) {
      fEvent = evt;
      fAngle.clear();
    }
    // Angle to the gun direction, from the first step ending on or beyond the plane through
    // the tank centre (origin) that is perpendicular to the gun direction.
    const int id = track->GetTrackID();
    if (!fAngle.count(id) && post->GetPosition().dot(gGunDir) >= 0.) {
      fAngle[id] = post->GetMomentumDirection().angle(gGunDir) / deg;
    }

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
    const auto it = fAngle.find(id);
    am->FillNtupleDColumn(8, it != fAngle.end() ? it->second : -1.);
    am->AddNtupleRow();
  }

private:
  int fEvent = -1;
  std::map<int, double> fAngle;  // trackID -> angle [deg]
};

class EventAction : public G4UserEventAction {
public:
  void BeginOfEventAction(const G4Event*) override { gSiHits.clear(); }
  void EndOfEventAction(const G4Event* event) override {
    RunAction::gX.clear(); RunAction::gY.clear(); RunAction::gZ.clear(); RunAction::gN.clear();
    RunAction::gTFirst.clear(); RunAction::gTMean.clear();
    for (const auto& [idx, n] : gSiHits) {
      const auto& p = gSiPos[idx];
      RunAction::gX.push_back(p.x() / mm);
      RunAction::gY.push_back(p.y() / mm);
      RunAction::gZ.push_back(p.z() / mm);
      RunAction::gN.push_back(n.n);
      RunAction::gTFirst.push_back(n.tFirst / ns);
      RunAction::gTMean.push_back(n.tSum / n.n / ns);
    }
    auto* am = G4AnalysisManager::Instance();
    am->FillNtupleIColumn(1, 0, event->GetEventID());
    // Column 7 follows eventID and the six vector columns
    const auto* vtx = event->GetPrimaryVertex();
    am->FillNtupleIColumn(1, 7, vtx && vtx->GetPrimary() ? vtx->GetPrimary()->GetPDGcode() : 0);
    am->AddNtupleRow(1);
  }
};

class ActionInitialization : public G4VUserActionInitialization {
public:
  void BuildForMaster() const override { SetUserAction(new RunAction()); }
  void Build() const override {
    SetUserAction(new PrimaryGenerator());
    SetUserAction(new RunAction());
    SetUserAction(new SteppingAction());
    SetUserAction(new EventAction());
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
  // Batch default is VRML2FILE; an interactive session defaults to an OpenGL window.
  // Macros read it via /control/getEnv. Set VIS_DRIVER to override.
  setenv("VIS_DRIVER", interactive ? "OGL" : "VRML2FILE", 0);
  // Pass argc=1 so the session type is not guessed from the macro name.
  G4UIExecutive* ui = interactive ? new G4UIExecutive(1, argv) : nullptr;

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
