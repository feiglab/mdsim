from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Optional

import numpy as np
from openmm import (
    CMMotionRemover,
    LangevinIntegrator,
    Platform,
    State,
    System,
    Vec3,
    XmlSerializer,
)
from openmm.app import (
    CharmmCrdFile,
    CharmmParameterSet,
    CharmmPsfFile,
    DCDReporter,
    ForceField,
    GromacsGroFile,
    GromacsTopFile,
    PDBFile,
    Simulation,
    StateDataReporter,
    element,
)
from openmm.app import (
    forcefield as ff,
)
from openmm.unit import (
    Quantity,
    kelvin,
    kilojoule,
    mole,
    nanometer,
    picoseconds,
)

from .__version__ import __version__

# --- Data containers ---------------------------------------------------------

aminoacids = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HSD",
    "HSE",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

nucleicacids = ["ADE", "CYT", "GUA", "URA", "THY"]

# --- Main class for COCOMO system --------------------------------------------


class MDSim:
    def __init__(
        self,
        *,
        structure=None,  # Structure/Model object read via PDBReader
        pdb=None,  # PDB file
        crd=None,  # CHARMM CRD file
        psf=None,  # CHARMM PSF file
        gro=None,  # GROMACS gro file
        ff=None,  # one or more XML force field files
        par=None,  # one or more CHARMM parameter files
        gmx=None,  # GROMACS topology file
        xml=None,  # System XML file
        positions=None,  # directly provide positions
        topology=None,  # directly provide topology
        restart=None,  # Restart XML with coordinates/velocities
        box=100,  # 100 or (50,20,40), nm
        nonbonded="PME",  # 'PME', 'LJPME', 'NoCutoff', 'CutoffPeriodic', 'CutoffNonPeriodic'
        cuton=1.0,  # nm
        cutoff=1.2,  # nm
        switching=True,  # 'openmm' (default), 'charmmm', True, False
        constraints=True,  # 'HBonds'  (default), 'AllBonds', 'HAngles'
        rigidwater=True,  # True, Fals
        dispcorr=False,  #
        hmass=None,  # hydrogen mass repartioning, True (3 amu) or give value
        removecmmotion=False,  # remove center of mass motion
    ):

        self.simulation = None

        self.topology = None
        self.positions = None

        self.set_pdb(pdb)
        self.set_psf(psf)
        self.set_crd(crd)
        self.set_charmmpar(par)
        self.set_gro(gro)
        self.set_gmxtop(gmx)
        self.set_xmlff(ff)

        if box:
            self.set_box(box)
        if positions:
            self.positions = positions
        if topology:
            self.topology = topology

        self.cuton = cuton * nanometer
        self.cutoff = cutoff * nanometer
        self.set_nonbonded(nonbonded)
        self.set_switching(switching)
        self.ewaldtol = 1e-05
        self.dispcorr = dispcorr

        self.rigidwater = rigidwater

        self.set_constraints(constraints)
        self.set_hydrogenmass(hmass)

        self.removecmmotion = removecmmotion

        if xml:
            self.read_system(xml)
            return

        self.fix_topology()

        self.setup_system()

    def set_nonbonded(self, nb):
        self.nonbonded = ff.PME
        if nb.lower() == "pme":
            self.nonbonded = ff.PME
        elif nb.lower() == "ljpme":
            self.nonbonded = ff.LJPME
        elif nb.lower() == "nocutoff":
            self.nonbonded = ff.NoCutoff
        elif nb.lower() == "cutoff":
            if self.box:
                self.nonbonded = ff.CutoffPeriodic
            else:
                self.nonbonded = ff.CutoffNonPeriodic
        elif nb.lower() == "cutoffperiodic":
            self.nonbonded = ff.CutoffPeriodic
        elif nb.lower() == "cutoffnonperiodic":
            self.nonbonded = ff.CutoffNonPeriodic

    def set_constraints(self, c):
        self.constraints = None
        self.flexcons = True
        if isinstance(c, bool) and c:
            self.constraints = ff.HBonds
            self.flexcons = False
        elif isinstance(c, str):
            if c.lower() == "hbonds":
                self.constraints = ff.HBonds
                self.flexcons = False
            elif c.lower() == "allbonds":
                self.constraints = ff.AllBonds
                self.flexcons = False
            elif c.lower() == "hangles":
                self.constraints = ff.HAngles
                self.flexcons = False

    def set_hydrogenmass(self, hm):
        self.hydrogenmass = None
        if isinstance(hm, bool) and hm:
            self.hydrogenmass = 3.0
        elif isinstance(hm, str):
            hmassval = float(hm)
            if hmassval > 1.0:
                self.hydrogenmass = hmassval

    def set_switching(self, sw):
        self.switching = None
        if isinstance(sw, bool) and sw:
            self.switching = "openmm"
        elif isinstance(sw, str):
            if sw.lower() == "openmm":
                self.switching = "openmm"
            elif sw.lower() == "charmm":
                self.switching = "charmm"
            else:
                self.switching = sw

    def set_pdb(self, fname):
        self.pdb = None
        if _is_readable_file(fname):
            self.pdb = PDBFile(fname)
            self.topology = self.pdb.topology
            self.positions = self.pdb.positions

    def set_psf(self, fname):
        self.psf = None
        if _is_readable_file(fname):
            self.psf = CharmmPsfFile(fname)
            self.topology = self.psf.topology

    def set_crd(self, fname):
        self.crd = None
        if _is_readable_file(fname):
            self.crd = CharmmCrdFile(fname)
            self.positions = self.crd.positions

    def set_gro(self, fname):
        self.gro = None
        if _is_readable_file(fname):
            self.gro = GromacsGroFile(fname)
            self.positions = self.gro.positions
            # box size?

    def set_gmxtop(self, fname):
        self.gmx = None
        if _is_readable_file(fname):
            self.gmx = GromacsTopFile(fname)
            self.topology = self.gmx.topology

    def set_xmlff(self, fname):
        self.ff = None
        if fname:
            if isinstance(fname, str):
                flist = [fname]
            else:
                flist = list(fname)

            plist = []
            for f in flist:
                if _is_readable_file(f):
                    plist.append(f)
            if plist:
                self.ff = ForceField(*plist)

    def set_charmmpar(self, fname):
        self.cpar = None
        if fname:
            if isinstance(fname, str):
                flist = [fname]
            else:
                flist = list(fname)

            plist = []
            for f in flist:
                if _is_readable_file(f):
                    plist.append(f)
            if plist:
                self.cpar = CharmmParameterSet(*plist)

    def fix_topology(self):
        if self.topology:
            for atom in self.topology.atoms():
                if atom.name == "SOD":
                    atom.element = element.sodium

    def setup_system(self):
        self.system = None
        if self.psf and self.cpar:
            if self.box:
                self.psf.setBox(
                    self.box[0] * nanometer, self.box[1] * nanometer, self.box[2] * nanometer
                )
            self.system = self.psf.createSystem(
                self.cpar,
                nonbondedMethod=self.nonbonded,
                nonbondedCutoff=self.cutoff,
                ewaldErrorTolerance=self.ewaldtol,
                switchDistance=None,
                constraints=self.constraints,
                flexibleConstraints=self.flexcons,
                rigidWater=self.rigidwater,
                removeCMMotion=self.removecmmotion,
                hydrogenMass=self.hydrogenmass,
            )
            print("created system from CHARMM PSF")
        elif self.topology and self.ff:
            self.system = self.ff.createSystem(
                self.topology,
                nonbondedMethod=self.nonbonded,
                nonbondedCutoff=self.cutoff,
                ewaldErrorTolerance=self.ewaldtol,
                switchDistance=None,
                constraints=self.constraints,
                rigidWater=self.rigidwater,
                useDispersionCorrection=self.dispcorr,
                removeCMMotion=self.removecmmotion,
                hydrogenMass=self.hydrogenmass,
            )
            print("created system from XML ForceField")
        elif self.gmx:
            self.system = self.gmx.createSystem(
                nonbondedMethod=self.nonbonded,
                nonbondedCutoff=self.cutoff,
                ewaldErrorTolerance=self.ewaldtol,
                switchDistance=None,
                constraints=self.constraints,
                rigidWater=self.rigidwater,
                useDispersionCorrection=self.dispcorr,
                removeCMMotion=self.removecmmotion,
                hydrogenMass=self.hydrogenmass,
            )
            print("created system from Gromacs topology")
        if self.system:
            for i, force in enumerate(self.system.getForces()):
                force.setForceGroup(i)
        else:
            self.system = System()

    def describe(self) -> str:
        return f"This is MDSim version {__version__}"

    def write_system(self, fname="system.xml"):
        with open(fname, "w") as file:
            file.write(XmlSerializer.serialize(self.system))

    def read_system(self, fname="system.xml"):
        with open(fname) as file:
            self.system = XmlSerializer.deserialize(file.read())

    def write_state(self, fname="state.xml"):
        if self.simulation is not None:
            self.simulation.saveState(fname)

    def read_state(self, fname="state.xml"):
        if self.simulation is not None:
            self.simulation.loadState(fname)

    def write_pdb(self, fname="state.pdb"):
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        with open(fname, "w") as f:
            PDBFile.writeFile(self.simulation.topology, positions, f)

    def get_potentialEnergy(self):
        if self.simulation is not None:
            return self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        else:
            return 0.0

    def get_energies(self):
        """
        Return per-force energies.

        - Ensures unique force groups (0–31) by reassigning duplicates.
        - Sorts forces in the order:
          HarmonicBondForce, HarmonicAngleForce, CustomBondForce,
          PeriodicTorsionForce, CustomTorsionForce, CMAPTorsionForce,
          NonbondedForce, CustomNonbondedForce, then any others.
        - Dict keys are 'group:ForceName' so the same name can appear multiple times.
        """
        # Desired order by class name
        priority = {
            "HarmonicBondForce": 0,
            "HarmonicAngleForce": 1,
            "CustomBondForce": 2,
            "PeriodicTorsionForce": 3,
            "CustomTorsionForce": 4,
            "CMAPTorsionForce": 5,
            "NonbondedForce": 6,
            "CustomNonbondedForce": 7,
        }

        # First pass: ensure unique force groups
        forces = list(self.system.getForces())
        used_groups = set()

        for frc in forces:
            g = frc.getForceGroup()
            if g in used_groups:
                # Find an unused group in [0, 31]
                for candidate in range(32):
                    if candidate not in used_groups:
                        frc.setForceGroup(candidate)
                        g = candidate
                        break
            used_groups.add(g)

        # Prepare sortable list with original indices as tie-breakers
        indexed_forces = []
        for idx, frc in enumerate(forces):
            cls_name = frc.__class__.__name__
            order = priority.get(cls_name, 100)  # unknown forces go last
            indexed_forces.append((order, idx, frc))

        # Sort by requested order, then original index
        indexed_forces.sort(key=lambda x: (x[0], x[1]))

        # Compute energies
        energies = {}
        for _, _, frc in indexed_forces:
            g = frc.getForceGroup()
            e = self._group_energy(self.simulation.context, g)
            name = frc.getName() or frc.__class__.__name__
            key = f"{name}({g})"
            energies[key] = e

        return energies

    def set_velocities(self, *, seed=None, newtemp=None):
        if self.simulation is not None:
            if newtemp is not None:
                temperature = newtemp * kelvin
            else:
                temperature = self.temperature
            if seed is None:
                seed = np.random.SeedSequence().entropy
            seed = int(seed) & 0x7FFFFFFF
            self.simulation.context.setVelocitiesToTemperature(temperature, seed)

    def minimize(self, *, nstep=1000, tol=0.001):
        if self.simulation is not None:
            tolerance = tol * kilojoule / (nanometer * mole)
            self.simulation.minimizeEnergy(tolerance=tolerance, maxIterations=nstep)

    def simulate(self, *, nstep=1000, nout=100, logfile="energy.log", dcdfile="traj.dcd"):
        if self.simulation is not None:
            dcd = DCDReporter(dcdfile, nout)
            self.simulation.reporters.append(dcd)
            log = StateDataReporter(
                logfile,
                nout,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=nstep,
                separator=" ",
            )
            self.simulation.reporters.append(log)
            self.simulation.step(nstep)

    def setup_simulation(
        self,
        *,
        temperature=298,
        gamma=0.01,
        tstep=0.001,
        resources="CPU",
        device=0,
        positions=None,
        restart=None,
    ) -> None:
        # temperature: in K
        # tstep: in ps
        # gamma: in 1/ps
        # resources: 'CPU' or 'CUDA'

        assert self.topology is not None, "need topology to be defined"
        assert self.system is not None, "need openMM system object to be defined"

        self.temperature = temperature * kelvin
        self.tstep = tstep * picoseconds
        self.gamma = gamma / picoseconds
        self.resources = resources

        self.integrator = LangevinIntegrator(self.temperature, self.gamma, self.tstep)
        self.platform = Platform.getPlatformByName(self.resources)
        self.simulation = None
        if self.resources == "CUDA":
            prop = dict(CudaPrecision="mixed", CudaDeviceIndex=str(device))
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, self.platform, prop
            )
        if self.resources == "CPU":
            self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)
        if restart is not None:
            self.read_state(restart)
        else:
            if positions is not None:
                self.positions = positions
            if self.positions is not None:
                self.simulation.context.setPositions(self.positions)

    def set_box(self, box) -> None:
        ax_nm, by_nm, cz_nm = self._normalize_box(box)

        a_sys = Vec3(ax_nm, 0.0, 0.0) * nanometer
        b_sys = Vec3(0.0, by_nm, 0.0) * nanometer
        c_sys = Vec3(0.0, 0.0, cz_nm) * nanometer

        # record on the class
        self.box: tuple[float, float, float] = (ax_nm, by_nm, cz_nm)
        self.box_vectors = (a_sys, b_sys, c_sys)

        # set on topology and system
        a_top = Vec3(ax_nm, 0.0, 0.0)
        b_top = Vec3(0.0, by_nm, 0.0)
        c_top = Vec3(0.0, 0.0, cz_nm)

        if self.topology is not None:
            self.topology.setPeriodicBoxVectors((a_top, b_top, c_top))

    #        self.system.setDefaultPeriodicBoxVectors(a_sys, b_sys, c_sys)

    def setup_forces(self) -> None:
        self.forces = {}
        if self.topology is not None:
            if self.removecmmotion:
                self.setupCMMotionRemover()
            self.forcemapping = self.assign_force_groups()

    def setupCMMotionRemover(self) -> None:
        if self.topology is not None and self.removecmmotion:
            force = CMMotionRemover()
            force.setName("cmmotion")
            self.forces["cmmotion"] = force
            self.system.addForce(force)

    def assign_force_groups(self):
        mapping = {}
        for i, frc in enumerate(self.system.getForces()):
            frc.setForceGroup(i % 32)
            name = frc.getName()
            if not name:
                name = frc.__class__.__name__
            mapping[frc.getForceGroup()] = (i, name)
        return mapping

    @staticmethod
    def _group_energy(context, group: int):
        mask = 1 << group
        st: State = context.getState(getEnergy=True, groups=mask)
        return st.getPotentialEnergy()

    def set_bonds(self):
        if self.topology is not None:
            for c in self.topology.chains():
                atm = []
                for r in c.residues():
                    for a in r.atoms():
                        atm.append(a)
                for i in range(len(atm) - 1):
                    self.topology.addBond(atm[i], atm[i + 1])

    @staticmethod
    def _normalize_box(box) -> tuple[float, float, float]:
        """
        Normalize user input to a 3-tuple of floats in Angstroms
        Accepts:
          - scalar number (int/float)
          - 3-sequence of numbers
          - openmm.unit.Quantity scalar/3-sequence with length units
        """
        # Quantity support (optional but handy)
        if isinstance(box, Quantity):
            # convert to nm and pull magnitude
            box_in_nm = box.value_in_unit(nanometer)
            if isinstance(box_in_nm, (int, float)):
                val = float(box_in_nm)
                return (val, val, val)
            # sequence quantity
            if isinstance(box_in_nm, Sequence) and len(box_in_nm) == 3:
                ax, by, cz = map(float, box_in_nm)
                return (ax, by, cz)
            raise TypeError("Quantity box must be scalar or length-3.")

        # Plain numeric
        if isinstance(box, (int, float)):
            val = float(box) / 10.0
            return (val, val, val)

        # Plain sequence
        if isinstance(box, Sequence) and len(box) == 3:
            ax, by, cz = box
            if not all(isinstance(v, (int, float)) for v in (ax, by, cz)):
                raise TypeError("Box tuple must contain numbers.")
            return (float(ax) / 10.0, float(by) / 10.0, float(cz) / 10.0)

        raise TypeError("box must be a number, a length-3 tuple, or a Quantity.")


def _is_readable_file(path: Optional[str]) -> bool:
    """Return True if `path` is a readable file; False for None or invalid types."""
    if not isinstance(path, str):
        return False
    return os.path.isfile(path) and os.access(path, os.R_OK)
