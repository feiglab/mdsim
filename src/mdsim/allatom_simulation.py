from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Optional

import mdtraj as md
import numpy as np
from openmm import (
    CMMotionRemover,
    CustomBondForce,
    CustomCentroidBondForce,
    CustomExternalForce,
    CustomNonbondedForce,
    LangevinIntegrator,
    MonteCarloBarostat,
    NonbondedForce,
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
    Topology,
    element,
)
from openmm.app import (
    forcefield as ff,
)
from openmm.unit import (
    Quantity,
    bar,
    kelvin,
    kilojoule,
    mole,
    nanometer,
    picoseconds,
    radian,
)

from .__version__ import __version__
from .molecule_data import Model, PDBReader, Structure

# --- Main class for COCOMO system --------------------------------------------


class MDSim:
    def __init__(
        self,
        *,
        model=None,  # Structure/Model object or file name read via PDBReader
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
        temperature=298,  # temperature: in K
        pressure=None,  # pressure in bar
        gamma=0.01,  # gamma in 1/ps
        tstep=0.001,  # time step in ps
        box=None,  # 100 or (50,20,40), Angstroms
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
        self.velocities = None
        self.box = None
        self.box_vectors = None
        self.stype = None

        self.set_psf(psf)
        self.set_gmxtop(gmx)
        self.set_pdb(pdb)
        self.set_model(model)
        self.set_crd(crd)
        self.set_gro(gro)
        self.set_charmmpar(par)
        self.set_xmlff(ff)
        self.read_restart(restart)

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
        self.ewaldtol = 5e-04
        self.dispcorr = dispcorr

        self.rigidwater = rigidwater

        self.set_constraints(constraints)
        self.set_hydrogenmass(hmass)

        self.removecmmotion = removecmmotion

        self.temperature = temperature * kelvin
        if pressure:
            self.pressure = pressure * bar
        else:
            self.pressure = None

        self.tstep = tstep * picoseconds
        self.gamma = gamma / picoseconds

        if xml:
            self.read_system(xml)
        else:
            self.fix_topology()
            self.setup_system()

    def setup_simulation(
        self,
        *,
        restart=None,
        positions=None,
        velocities=None,
        box=None,
        resources="CPU",
        device=0,
        tstep=None,
        gamma=None,
        temperature=None,
    ):
        self.resources = resources

        if restart:
            self.read_restart(restart)
        if positions:
            self.positions = positions
        if velocities:
            self.velocities = velocities
        if box:
            self.set_box(box)
        if tstep:
            self.tstep = tstep * picoseconds
        if gamma:
            self.gamma = gamma / picoseconds
        if temperature:
            self.temperature = temperature * kelvin

        if not self.topology:
            self.set_dummy_topology()

        self.integrator = LangevinIntegrator(self.temperature, self.gamma, self.tstep)
        self.platform = Platform.getPlatformByName(self.resources)

        if self.resources == "CUDA":
            prop = dict(CudaPrecision="mixed", CudaDeviceIndex=str(device))
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, self.platform, prop
            )
        if self.resources == "CPU":
            self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)

        if self.positions:
            self.simulation.context.setPositions(self.positions)
        if self.velocities:
            self.simulation.context.setVelocities(self.velocities)
        else:
            self.set_velocities()
        if self.box_vectors:
            a, b, c = self.box_vectors
            self.simulation.context.setPeriodicBoxVectors(a, b, c)

    def get_positions(self):
        if self.simulation:
            return self.simulation.context.getState(getPositions=True).getPositions()

    def get_velocities(self):
        if self.simulation:
            return self.simulation.context.getState(getVelocities=True).getVelocities()

    def set_nonbonded(self, nb):
        self.nonbonded = ff.PME
        if nb.lower() == "pme":
            self.nonbonded = ff.PME
        elif nb.lower() == "ljpme":
            self.nonbonded = ff.LJPME
        elif nb.lower() == "nocutoff":
            self.nonbonded = ff.NoCutoff
        elif nb.lower() == "cutoff":
            if self.box_vectors:
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

    def set_model(self, model):
        self.model = None
        if isinstance(model, Structure):
            self.model = model[0]
        elif isinstance(model, Model):
            self.model = model
        elif isinstance(model, str):
            self.model = PDBReader(model)[0]
        if self.model:
            if not self.positions:
                self.positions = self.model.positions()
            if not self.topology:
                self.topology = self.model.topology()

    def set_pdb(self, fname):
        self.pdb = None
        if _is_readable_file(fname):
            self.pdb = PDBFile(fname)
            if not self.positions:
                self.positions = self.pdb.positions
            if not self.topology:
                self.topology = self.pdb.topology

    def set_psf(self, fname):
        self.psf = None
        if _is_readable_file(fname):
            self.psf = CharmmPsfFile(fname)
            if not self.topology:
                self.topology = self.psf.topology

    def set_crd(self, fname):
        self.crd = None
        if _is_readable_file(fname):
            self.crd = CharmmCrdFile(fname)
            if not self.positions:
                self.positions = self.crd.positions

    def set_gro(self, fname):
        self.gro = None
        if _is_readable_file(fname):
            self.gro = GromacsGroFile(fname)
            if not self.positions:
                self.positions = self.gro.positions
            # set box from GRO if present
            try:
                box_vec = self.gro.getUnitCellDimensions()  # Vec3 * unit
            except AttributeError:
                box_vec = None
            if box_vec is not None:
                # convert to Quantity in nm and feed _normalize_box
                self.set_box(box_vec)

    def set_gmxtop(self, fname):
        self.gmx = None
        if _is_readable_file(fname):
            self.gmx = GromacsTopFile(fname)
            if not self.topology:
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

    def read_restart(self, fname):
        if _is_readable_file(fname):
            with open(fname) as f:
                state = XmlSerializer.deserialize(f.read())
            self.positions = state.getPositions()
            self.velocities = state.getVelocities()
            self.box_vectors = state.getPeriodicBoxVectors()

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
            self.stype = "charmm"
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
            self.stype = "xmlff"
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
            self.stype = "gmx"
        if self.system:
            self.set_switching_function()
            self.set_barostat()
            self.set_force_groups()
        else:
            self.system = System()

    def set_switching_function(self):
        if self.switching and self.cuton < self.cutoff and self.cuton > 0 * nanometer:
            if self.switching == "openmm":
                flist = []
                for i, force in enumerate(self.system.getForces()):
                    if isinstance(force, NonbondedForce):
                        flist.append(force)
                    if isinstance(force, CustomNonbondedForce):
                        flist.append(force)
                if flist:
                    for f in flist:
                        f.setUseSwitchingFunction(True)
                        f.setSwitchingDistance(self.cuton)
                        f.setCutoffDistance(self.cutoff)
            if self.switching == "charmm":
                for i, force in enumerate(self.system.getForces()):
                    name = force.getName() or force.__class__.__name__
                    if self.stype and self.stype == "xmlff" and name == "LennardJones":
                        ljswitch = "step(Ron-r)*(ba*tr6*tr6-bb*tr6+bb*oo3-ba*oo6) \
                          +step(r-Ron)*step(Roff-r)*(cr12*rj6-cr6*rj3) \
                          -step(r-Ron)*step(Ron-r)*(cr12*rj6-cr6*rj3); \
                          cr6=bb*od3*rj3; cr12=ba*od6*rj6; \
                          rj3=r3-rc3; rj6=tr6-rc6; r3=r1*tr2; r1=sqrt(tr2); \
                          tr6=tr2*tr2*tr2; tr2=1.0/s2; s2=r*r; \
                          bb = bcoef(type1,type2); ba = acoef(type1,type2); \
                          oo3=rc3/on3; oo6=rc6/on6; od3=off3/(off3-on3); od6=off6/(off6-on6); \
                          rc3=1.0/off3; on6=on3*on3; on3=c2onnb*Ron; \
                          rc6=1.0/off6; off6=off3*off3; off3=c2ofnb*Roff; \
                          c2ofnb=Roff*Roff; c2onnb=Ron*Ron"
                        force.addGlobalParameter("Ron", self.cuton)
                        force.addGlobalParameter("Roff", self.cutoff)
                        force.setEnergyFunction(ljswitch)

                    if self.stype and (
                        (self.stype == "xmlff" and name == "LennardJones14")
                        or (self.stype == "charmm" and name == "NonbondedForce")
                        or (self.stype == "gmx" and name == "NonbondedForce")
                    ):
                        ljswitch = "step(Ron-r)*(ba*tr6*tr6-bb*tr6+bb*oo3-ba*oo6) \
                          +step(r-Ron)*step(Roff-r)*(cr12*rj6-cr6*rj3) \
                          -step(r-Ron)*step(Ron-r)*(cr12*rj6-cr6*rj3); \
                          cr6=bb*od3*rj3; cr12=ba*od6*rj6; \
                          rj3=r3-rc3; rj6=tr6-rc6; r3=r1*tr2; r1=sqrt(tr2); \
                          tr6=tr2*tr2*tr2; tr2=1.0/s2; s2=r*r; \
                          bb=4.0*epsilon*sigma^6; ba=4.0*epsilon*sigma^12; \
                          oo3=rc3/on3; oo6=rc6/on6; od3=off3/(off3-on3); od6=off6/(off6-on6); \
                          rc3=1.0/off3; on6=on3*on3; on3=c2onnb*Ron; \
                          rc6=1.0/off6; off6=off3*off3; off3=c2ofnb*Roff; \
                          c2ofnb=Roff*Roff; c2onnb=Ron*Ron"

                        if self.stype == "charmm" or self.stype == "gmx":
                            f = CustomBondForce(ljswitch)
                            f.addGlobalParameter("Ron", self.cuton)
                            f.addGlobalParameter("Roff", self.cutoff)
                            f.addPerBondParameter("sigma")
                            f.addPerBondParameter("epsilon")
                            for j in range(force.getNumExceptions()):
                                a1, a2, chg, sig, eps = force.getExceptionParameters(j)
                                force.setExceptionParameters(
                                    j, a1, a2, chg, 0.0, 0.0
                                )  # zero sig/eps
                                f.addBond(a1, a2, [sig, eps])
                            self.system.addForce(f)
                        else:
                            force.addGlobalParameter("Ron", self.cuton)
                            force.addGlobalParameter("Roff", self.cutoff)
                            force.setEnergyFunction(ljswitch)

                    if self.stype and (
                        (self.stype == "gmx" and name == "LennardJonesForce")
                        or (self.stype == "charmm" and name == "CustomNonbondedForce")
                    ):
                        ljswitch = "step(Ron-r)*(ba*tr6*tr6-bb*tr6+bb*oo3-ba*oo6) \
                          +step(r-Ron)*step(Roff-r)*(cr12*rj6-cr6*rj3) \
                          -step(r-Ron)*step(Ron-r)*(cr12*rj6-cr6*rj3); \
                          cr6=bb*od3*rj3; cr12=ba*od6*rj6; \
                          rj3=r3-rc3; rj6=tr6-rc6; r3=r1*tr2; r1=sqrt(tr2); \
                          tr6=tr2*tr2*tr2; tr2=1.0/s2; s2=r*r; \
                          bb = bcoef(type1,type2); ba=baa*baa; baa=acoef(type1,type2); \
                          oo3=rc3/on3; oo6=rc6/on6; od3=off3/(off3-on3); od6=off6/(off6-on6); \
                          rc3=1.0/off3; on6=on3*on3; on3=c2onnb*Ron; \
                          rc6=1.0/off6; off6=off3*off3; off3=c2ofnb*Roff; \
                          c2ofnb=Roff*Roff; c2onnb=Ron*Ron"
                        force.addGlobalParameter("Ron", self.cuton)
                        force.addGlobalParameter("Roff", self.cutoff)
                        force.setEnergyFunction(ljswitch)

    def set_position_restraint(
        self, *, selection="name CA", atomlist=None, k=100.0, positions=None
    ):
        """
        Apply PBC-aware positional restraints to a set of atoms.

        Parameters
        ----------
        selection : str
            MDTraj selection string (ignored if `atomlist` is provided).
        atomlist : sequence of int or openmm.app.Atom
            Atoms to restrain, given as indices or Atom objects.
        k : float
            Force constant in kJ/mol/nm^2.
        positions : reference positions
        """
        if self.system is None:
            raise RuntimeError("System has not been created yet.")

        # Determine atom indices
        if atomlist is not None:
            # Accept list of ints or list of Atom objects
            if len(atomlist) == 0:
                return
            first = atomlist[0]
            if isinstance(first, int):
                indices = list(atomlist)
            else:
                # assume OpenMM Atom objects
                indices = [a.index for a in atomlist]
        else:
            if self.topology is None:
                raise RuntimeError("Topology is required for selection-based restraints.")
            md_top = md.Topology.from_openmm(self.topology)
            indices = md_top.select(selection).tolist()

        if not indices:
            return  # nothing to restrain

        # Get reference positions (in nm)
        if positions is not None:
            pos = positions
        elif self.positions is not None:
            pos = self.positions
        elif self.simulation is not None:
            pos = self.simulation.context.getState(getPositions=True).getPositions()
        else:
            raise RuntimeError("No positions available to define restraint reference points.")

        force = CustomExternalForce("0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", k * kilojoule / (nanometer**2 * mole))
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        # Convert positions to nm if they are a Quantity
        if hasattr(pos, "value_in_unit"):
            # pos is a Quantity (e.g. from State.getPositions())
            pos_nm = pos.value_in_unit(nanometer)
        else:
            # pos is already a list/array of Vec3 or xyz triples in nm
            pos_nm = pos

        for idx in indices:
            p = pos_nm[idx]
            # p can be a Vec3 or a length-3 iterable of floats
            if hasattr(p, "x"):
                x0, y0, z0 = p.x, p.y, p.z
            else:
                x0, y0, z0 = p[0], p[1], p[2]
            force.addParticle(idx, [x0, y0, z0])

        force.setName("PositionalRestraints")
        self.system.addForce(force)

    def set_barostat(self, pressure=None, temperature=None):
        if self.system:
            if pressure:
                self.pressure = pressure * bar
            if temperature:
                self.temperature = temperature * kelvin
            if self.temperature and self.pressure:
                barostat = MonteCarloBarostat(self.pressure, self.temperature)
                self.system.addForce(barostat)

    def set_umbrella_xyz_distance(self, groupa, groupb, *, direction="x", target=0.0, k=10.0):
        if self.system:
            bias = f"0.5 * uk_{direction} * ((abs({direction}2 - {direction}1) - target)^2)"
            force = CustomCentroidBondForce(2, bias)
            force.addPerBondParameter("target")  # target distance (nm)
            force.addGlobalParameter(f"uk_{direction}", k * kilojoule / mole / nanometer**2)
            force.addGroup(groupa)
            force.addGroup(groupb)
            force.addBond([0, 1], [target * nanometer])
            force.setName(f"Umbrella_{direction}")
            self.system.addForce(force)

    def update_umbrella_xyz_distance(self, direction="x", k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter(
                f"uk_{direction}", k * kilojoule / mole / nanometer**2
            )

    def set_umbrella_distance(self, groupa, groupb, *, target=0.0, k=10.0, periodic=False):
        if self.system:
            bias = "0.5 * uk_dist * ((distance(g1,g2) - target)^2)"
            force = CustomCentroidBondForce(2, bias)
            force.addPerBondParameter("target")  # target distance (nm)
            force.addGlobalParameter("uk_dist", k * kilojoule / mole / nanometer**2)
            force.addGroup(groupa)
            force.addGroup(groupb)
            force.addBond([0, 1], [target * nanometer])
            if self.box_vectors and periodic:
                force.setUsesPeriodicBoundaryConditions(True)
            force.setName("Umbrella_distance")
            self.system.addForce(force)

    def update_umbrella_distance(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_dist", k * kilojoule / mole / nanometer**2)

    def _compute_com(self, group, positions_nm):
        """Compute mass-weighted COM of `group` using positions in nm."""
        masses = []
        coords = []
        for idx in group:
            m = self.system.getParticleMass(idx)
            masses.append(m._value)
            p = positions_nm[idx]
            if hasattr(p, "x"):
                coords.append([p.x, p.y, p.z])
            else:
                coords.append([p[0], p[1], p[2]])
        masses = np.asarray(masses, dtype=float)
        coords = np.asarray(coords, dtype=float)
        m_tot = masses.sum()
        if m_tot == 0.0:
            raise ValueError("Total mass of group is zero; cannot compute COM.")
        com = (masses[:, None] * coords).sum(axis=0) / m_tot
        return com  # (x, y, z) in nm

    def set_umbrella_com(
        self,
        group,
        *,
        k=10.0,
        target=None,  # single or per-group; see docstring
        periodic=False,
    ):
        """
        Restrain the COM of one or more groups to fixed points.

        Parameters
        ----------
        group
            Either:
              - sequence[int]: one group of atom indices, or
              - sequence[sequence[int]]: multiple groups, each a sequence of atom indices.
        k
            Force constant in kJ/mol/nm^2 (shared for all groups).
        target
            - None:
                For each group, target is taken as its current COM.
            - Single (x,y,z) in nm (tuple/list) or 3-element Quantity:
                Same target used for all groups.
            - Sequence of length 1 or n_groups:
                Per-group targets; each element may be:
                  * None  -> use current COM of that group
                  * (x,y,z) in nm (tuple/list)
                  * 3-element Quantity with length units
        periodic
            If True and a periodic box is defined, enable PBC on the bias.
        """
        from collections.abc import Sequence

        if self.system is None:
            raise RuntimeError("System has not been created yet.")

        # --- normalize groups to a list of lists of ints --------------------
        def _is_int_like(x):
            return isinstance(x, int)

        if not isinstance(group, Sequence) or len(group) == 0:
            return

        if _is_int_like(group[0]):
            # single group: [i,j,k,...]
            groups = [list(group)]
        else:
            # multiple groups: [[...], [...], ...]
            groups = [list(g) for g in group]

        if not groups:
            return

        n_groups = len(groups)

        # --- positions in nm (computed lazily, only if needed) -------------
        pos_nm = None

        def _get_pos_nm():
            nonlocal pos_nm
            if pos_nm is not None:
                return pos_nm

            if self.positions is not None:
                pos = self.positions
            elif self.simulation is not None:
                pos = self.simulation.context.getState(getPositions=True).getPositions()
            else:
                raise RuntimeError("No positions available to define COM restraint reference.")

            if hasattr(pos, "value_in_unit"):
                pos_nm_local = pos.value_in_unit(nanometer)
            else:
                pos_nm_local = pos  # assume already in nm

            pos_nm = pos_nm_local
            return pos_nm

        # --- helpers --------------------------------------------------------
        def _norm_xyz(t):
            """Return (x,y,z) in nm as floats from Quantity or 3-sequence."""
            if hasattr(t, "value_in_unit"):
                arr = t.value_in_unit(nanometer)
                return float(arr[0]), float(arr[1]), float(arr[2])
            # assume 3-sequence of numbers
            return float(t[0]), float(t[1]), float(t[2])

        def _is_scalar_xyz(seq):
            """Heuristic: 3 non-sequence elements -> treat as single xyz."""
            if not isinstance(seq, Sequence):
                return False
            if len(seq) != 3:
                return False
            for v in seq:
                if isinstance(v, Sequence) and not hasattr(v, "value_in_unit"):
                    return False
            return True

        # --- build per-group reference coordinates --------------------------
        xyz_list = []

        if target is None:
            # All targets from current COMs
            pos_nm = _get_pos_nm()
            for g in groups:
                xyz_list.append(self._compute_com(g, pos_nm))
        else:
            # target provided; could be:
            # - Quantity -> same for all groups
            # - (x,y,z) -> same for all groups
            # - sequence of per-group entries
            if hasattr(target, "value_in_unit") or _is_scalar_xyz(target):
                base = _norm_xyz(target)
                xyz_list = [base for _ in range(n_groups)]
            else:
                # Treat as per-group target list
                if not isinstance(target, Sequence):
                    raise TypeError(
                        "target must be None, a single (x,y,z)/Quantity, "
                        "or a sequence of per-group targets."
                    )

                # allow broadcasting: length 1 -> repeat for all groups
                if len(target) == 1 and n_groups > 1:
                    per_group = list(target) * n_groups
                else:
                    if len(target) != n_groups:
                        raise ValueError(
                            "Per-group target sequence length must be 1 or match "
                            "the number of groups."
                        )
                    per_group = list(target)

                for g, t in zip(groups, per_group):
                    if t is None:
                        pos_nm = _get_pos_nm()
                        xyz_list.append(self._compute_com(g, pos_nm))
                    else:
                        xyz_list.append(_norm_xyz(t))

        # --- define the CustomCentroidBondForce -----------------------------
        bias = "0.5 * uk_com * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        force = CustomCentroidBondForce(1, bias)
        force.addGlobalParameter("uk_com", k * kilojoule / (mole * nanometer**2))
        force.addPerBondParameter("x0")
        force.addPerBondParameter("y0")
        force.addPerBondParameter("z0")

        # add groups
        group_ids = []
        for g in groups:
            gid = force.addGroup(g)
            group_ids.append(gid)

        # add one bond per group
        for gid, (x0, y0, z0) in zip(group_ids, xyz_list):
            force.addBond([gid], [x0, y0, z0])

        if self.box_vectors and periodic:
            force.setUsesPeriodicBoundaryConditions(True)

        force.setName("Umbrella_COM")
        self.system.addForce(force)

    def update_umbrella_com(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_com", k * kilojoule / (mole * nanometer**2))

    def set_umbrella_angle_norm(
        self,
        groupa,
        groupa1,
        groupa2,
        groupb,
        groupb1,
        groupb2,
        *,
        target=np.radians(0),
        k=10.0,
    ):
        """
        Harmonic umbrella on the angle between two plane normals (groups A and B).

        Angle is in radians; target is in radians; k is in kJ/mol/rad^2.
        """
        if self.system:
            bias = (
                # harmonic in (angle - target)
                "0.5 * uk_angle_norm * (acos(cosang) - target)^2;"
                # numerically safe cosine between normals: clamp to [-1, 1]
                "cosang = min(1.0, max(-1.0, dotAB/(magA*magB)));"
                "dotAB = nxA_tmp*nxB_tmp + nyA_tmp*nyB_tmp + nzA_tmp*nzB_tmp;"
                "magA = sqrt(nxA_tmp^2 + nyA_tmp^2 + nzA_tmp^2);"
                "magB = sqrt(nxB_tmp^2 + nyB_tmp^2 + nzB_tmp^2);"
                # cross products nA = vA1 × vA2, nB = vB1 × vB2
                "nxA_tmp = vA1y*vA2z - vA1z*vA2y;"
                "nyA_tmp = vA1z*vA2x - vA1x*vA2z;"
                "nzA_tmp = vA1x*vA2y - vA1y*vA2x;"
                "nxB_tmp = vB1y*vB2z - vB1z*vB2y;"
                "nyB_tmp = vB1z*vB2x - vB1x*vB2z;"
                "nzB_tmp = vB1x*vB2y - vB1y*vB2x;"
                # plane A vectors (groups 1,2,3)
                "vA1x = x2 - x1;"
                "vA1y = y2 - y1;"
                "vA1z = z2 - z1;"
                "vA2x = x3 - x1;"
                "vA2y = y3 - y1;"
                "vA2z = z3 - z1;"
                # plane B vectors (groups 4,5,6)
                "vB1x = x5 - x4;"
                "vB1y = y5 - y4;"
                "vB1z = z5 - z4;"
                "vB2x = x6 - x4;"
                "vB2y = y6 - y4;"
                "vB2z = z6 - z4;"
            )

            force = CustomCentroidBondForce(6, bias)
            force.addPerBondParameter("target")  # radians
            force.addGlobalParameter("uk_angle_norm", k * kilojoule / (mole * radian**2))

            # group order: (A0, A1, A2, B0, B1, B2)
            force.addGroup(groupa)
            force.addGroup(groupa1)
            force.addGroup(groupa2)
            force.addGroup(groupb)
            force.addGroup(groupb1)
            force.addGroup(groupb2)

            force.addBond([0, 1, 2, 3, 4, 5], [target * radian])

            force.setName("Umbrella_angle_norm")
            self.system.addForce(force)

    def update_umbrella_angle_norm(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_angle_norm", k * kilojoule / mole / radian**2)

    def set_umbrella_dihedral(
        self,
        groupa,
        groupb,
        groupc,
        groupd,
        target=0.0,
        k=10.0,
    ):
        """
        Harmonic umbrella on the dihedral angle between four centroids.

        The minimum is at the dihedral = target (in radians), using a 2π-periodic
        quadratic distance: we choose the smallest of (Δ, Δ+2π, Δ-2π).

        Parameters
        ----------
        groupa, groupa1, groupb, groupb1 : sequence[int]
            Atom indices for the four centroid groups.
        target : float
            Target dihedral angle in radians.
        k : float
            Force constant in kJ/mol/rad^2.
        """
        if not self.system:
            return

        bias = (
            # periodic quadratic in the dihedral difference
            "0.5 * uk_dihedral * "
            "min((d - target)^2, min((d - target + 2*pi)^2, (d - target - 2*pi)^2));"
            "pi=acos(-1);"
            # d is the dihedral in radians
            "d = dihedral(g1, g2, g3, g4)"
        )

        force = CustomCentroidBondForce(4, bias)
        force.addPerBondParameter("target")  # radians
        force.addGlobalParameter("uk_dihedral", k * kilojoule / (mole * radian**2))

        # Order: (g1, g2, g3, g4)
        force.addGroup(groupa)
        force.addGroup(groupb)
        force.addGroup(groupc)
        force.addGroup(groupd)

        force.addBond([0, 1, 2, 3], [target * radian])

        force.setName("Umbrella_dihedral")
        self.system.addForce(force)

    def update_umbrella_dihedral(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_dihedral", k * kilojoule / mole / radian**2)

    def set_umbrella_angle(
        self,
        groupa,
        groupb,
        groupc,
        target=np.pi / 2.0,
        k=10.0,
    ):
        """
        Harmonic umbrellas on an angle defined by centroid triplets.

        Angle is in radians; restrained to `target`.

        The angle is:
          - angle(groupa,  groupb,  groupc)
        """
        if not self.system:
            return

        bias = "0.5 * uk_angle * (angle(g1, g2, g3) - target)^2"

        force = CustomCentroidBondForce(3, bias)
        force.addPerBondParameter("target")  # radians
        force.addGlobalParameter("uk_angle", k * kilojoule / (mole * radian**2))

        # group indices: 0=groupa, 1=groupb, 2=groupc
        force.addGroup(groupa)
        force.addGroup(groupb)
        force.addGroup(groupc)

        force.addBond([0, 1, 2], [target * radian])

        force.setName("Umbrella_angle")
        self.system.addForce(force)

    def update_umbrella_angle(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_angle", k * kilojoule / mole / radian**2)

    def set_force_groups(self):
        if self.system:
            for i, force in enumerate(self.system.getForces()):
                force.setForceGroup(i)

    def set_dummy_topology(self):
        if not self.topology and self.system:
            n_atoms = self.system.getNumParticles()
            top = Topology()
            chain = top.addChain()
            res = top.addResidue("DUM", chain)
            for i in range(n_atoms):
                top.addAtom("C", element.carbon, res)
            self.topology = top

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
        positions = self.get_positions()
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

        forces = list(self.system.getForces())

        indexed_forces = []
        for idx, frc in enumerate(forces):
            cls_name = frc.__class__.__name__
            order = priority.get(cls_name, 100)
            indexed_forces.append((order, idx, frc))

        indexed_forces.sort(key=lambda x: (x[0], x[1]))

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

    def simulate(self, *, nstep=1000, nout=1000, logfile=None, dcdfile=None):
        if self.simulation is not None:
            if dcdfile:
                dcd = DCDReporter(dcdfile, nout)
                self.simulation.reporters.append(dcd)
            if logfile:
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
        Normalize user input to a 3-tuple of floats in nm
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
