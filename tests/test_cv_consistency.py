from __future__ import annotations

from openmm import Context, System, Vec3, VerletIntegrator, unit

from mdsim.allatom_simulation import MDSim
from mdsim.molecule_data import Atom, Model


def _build_model_and_system(coords_nm: list[tuple[float, float, float]]):
    """
    coords_nm: list of (x, y, z) in nm, same ordering for Model and System.
    Model stores Å; System uses nm.
    """
    # Build Model with coords in Å
    atoms: list[Atom] = []
    for i, (x_nm, y_nm, z_nm) in enumerate(coords_nm):
        atoms.append(
            Atom(
                serial=i + 1,
                name=f"C{i+1}",
                element="C",
                resname="RES",
                chain="A",
                resnum=1,
                x=10.0 * x_nm,  # nm -> Å
                y=10.0 * y_nm,
                z=10.0 * z_nm,
                seg="A",
            )
        )

    model = Model(model_id=1)
    model.atoms = atoms  # we don't need chains/residues for these CVs

    # Build bare System with equal masses
    system = System()
    for _ in coords_nm:
        system.addParticle(12.0 * unit.amu)

    # Positions in nm for OpenMM
    positions = unit.Quantity(
        [Vec3(x, y, z) for (x, y, z) in coords_nm],
        unit.nanometer,
    )

    return model, system, positions


def _make_mdsim(system: System) -> MDSim:
    """
    Minimal MDSim instance that only provides .system (no __init__ side effects).
    """
    mds = object.__new__(MDSim)
    mds.system = system
    # Only some umbrellas care about box_vectors; set to None for these tests.
    mds.box_vectors = None
    return mds


def _energy_kjmol(system: System, positions):
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_distance_matches_openmm():
    # Two points 1.234 nm apart along x
    coords_nm = [
        (0.0, 0.0, 0.0),
        (1.234, 0.0, 0.0),
    ]
    model, system, positions = _build_model_and_system(coords_nm)
    mds = _make_mdsim(system)

    group_a = [0]
    group_b = [1]

    d_q = model.distance(group_a, group_b)
    d_nm = d_q.value_in_unit(unit.nanometer)

    # Add umbrella with target set to Model distance
    mds.set_umbrella_distance(group_a, group_b, target=d_nm, k=1.0, periodic=False)

    e = _energy_kjmol(system, positions)
    assert abs(e) < 1.0e-6


def test_angle_norm_matches_openmm():
    # Plane A: (0,0,0)-(1,0,0)-(0,1,0) -> normal ~ +z
    # Plane B: (0,0,0)-(1,0,0)-(0,0,1) -> normal ~ ±y
    # Angle between normals = 90°.
    coords_nm = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.0, 1.0, 0.0),  # 2
        (0.0, 0.0, 1.0),  # 3
    ]
    model, system, positions = _build_model_and_system(coords_nm)
    mds = _make_mdsim(system)

    group_a = [0]
    group_a1 = [1]
    group_a2 = [2]
    group_b = [0]
    group_b1 = [1]
    group_b2 = [3]

    theta_q = model.angle_norm(
        group_a,
        group_a1,
        group_a2,
        group_b,
        group_b1,
        group_b2,
    )
    theta = theta_q.value_in_unit(unit.radian)

    mds.set_umbrella_angle_norm(
        group_a,
        group_a1,
        group_a2,
        group_b,
        group_b1,
        group_b2,
        target=theta,
        k=1.0,
    )

    e = _energy_kjmol(system, positions)
    assert abs(e) < 1.0e-6


def test_dihedral_matches_openmm():
    # Non-planar four-point set
    coords_nm = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (1.0, 1.0, 1.0),  # 3
    ]
    model, system, positions = _build_model_and_system(coords_nm)
    mds = _make_mdsim(system)

    group_a = [0]
    group_b = [1]
    group_c = [2]
    group_d = [3]

    phi_q = model.dihedral(group_a, group_b, group_c, group_d)
    phi = phi_q.value_in_unit(unit.radian)

    mds.set_umbrella_dihedral(
        group_a,
        group_b,
        group_c,
        group_d,
        target=phi,
        k=1.0,
    )

    e = _energy_kjmol(system, positions)
    assert abs(e) < 1.0e-6


def test_angle_matches_openmm():
    # Simple 90° angle at atom 1: 0-1-2
    coords_nm = [
        (1.0, 0.0, 0.0),  # 0
        (0.0, 0.0, 0.0),  # 1
        (0.0, 1.0, 0.0),  # 2
    ]
    model, system, positions = _build_model_and_system(coords_nm)
    mds = _make_mdsim(system)

    group_a = [0]
    group_b = [1]
    group_c = [2]

    theta_q = model.angle(group_a, group_b, group_c)
    # Model.angle currently returns a Quantity directly
    theta = theta_q.value_in_unit(unit.radian)

    mds.set_umbrella_angle(
        group_a,
        group_b,
        group_c,
        target=theta,
        k=1.0,
    )

    e = _energy_kjmol(system, positions)
    assert abs(e) < 1.0e-6
