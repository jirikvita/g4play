G4BIN ?= /home/qitek/install/geant4-v11.1.1-install/bin
G4CONFIG := $(G4BIN)/geant4-config

TARGET := waterCherenkovSim
SRC := waterCherenkovSim.cpp

CXXFLAGS += $(shell $(G4CONFIG) --cflags)
LDLIBS += $(shell $(G4CONFIG) --libs)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

# Run with: source $(G4BIN)/geant4.sh first so data sets and libraries are found
clean:
	rm -f $(TARGET)

.PHONY: all clean
