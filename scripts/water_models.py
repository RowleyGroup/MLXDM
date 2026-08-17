"""
Rigid 3-point water model definitions shared by build_water_ball.py and
load_water_ball_openmm.py.

All numeric parameters below were taken verbatim from OpenMM's own bundled
force field XML files (charges, Lennard-Jones sigma/epsilon in nm/kJ-mol,
bond length in nm, angle in radians -- i.e. OpenMM's native units) so there
is a single, checkable source for each model rather than hand-copied
numbers:
    TIP3P    -> openmm/app/data/charmm36/water.xml
    OPC3     -> openmm/app/data/amber14/opc3.xml
    TIP3P-FB -> openmm/app/data/amber14/tip3pfb.xml

References:
    TIP3P:    Jorgensen et al., J. Chem. Phys. 1983, 79, 926
    OPC3:     Izadi & Onufriev, J. Chem. Phys. 2016, 145, 074501
              ("Accuracy limit of rigid 3-point water models")
    TIP3P-FB: Wang, Martinez & Pande, J. Phys. Chem. Lett. 2014, 5, 1885
              (the ForceBalance-refit TIP3P)
"""

import math
from dataclasses import dataclass

KCAL_PER_KJ = 1.0 / 4.184


@dataclass(frozen=True)
class WaterModel:
    key: str
    display_name: str
    reference: str
    resname: str      # CHARMM RESI / PDB resName (<=4 chars)
    atom_o: str        # oxygen atom name
    type_o: str         # CHARMM atom type for oxygen
    type_h: str
    mass_o: float        # amu
    mass_h: float
    q_o: float             # elementary charge
    q_h: float
    r_oh: float              # Angstrom
    angle_hoh_deg: float
    sigma_o_nm: float          # OpenMM/native units
    epsilon_o_kjmol: float
    sigma_h_nm: float
    epsilon_h_kjmol: float
    bond_k_kjmolnm2: float      # HarmonicBondForce k (unused under rigid/SHAKE water)
    angle_k_kjmolrad2: float     # HarmonicAngleForce k (unused under rigid/SHAKE water)

    @property
    def d_hh(self) -> float:
        """H-H distance implied by r_oh/angle_hoh_deg, in Angstrom."""
        return 2 * self.r_oh * math.sin(math.radians(self.angle_hoh_deg) / 2)

    @property
    def rmin2_o(self) -> float:
        """CHARMM-style Rmin/2 for oxygen, in Angstrom."""
        return self.sigma_o_nm * 10 * 2 ** (1 / 6) / 2

    @property
    def rmin2_h(self) -> float:
        return self.sigma_h_nm * 10 * 2 ** (1 / 6) / 2

    @property
    def epsilon_o_kcal(self) -> float:
        return self.epsilon_o_kjmol * KCAL_PER_KJ

    @property
    def epsilon_h_kcal(self) -> float:
        return self.epsilon_h_kjmol * KCAL_PER_KJ

    @property
    def bond_k_kcal_A2(self) -> float:
        # kJ/mol/nm^2 -> kcal/mol/A^2, and /2 because OpenMM's HarmonicBondForce
        # uses E = 1/2 k (r-r0)^2 while a CHARMM parameter file's printed Kb
        # already has that 1/2 folded in (E = Kb (b-b0)^2). Cross-checked
        # against the real CHARMM par_water_ions.str TIP3P value (Kb=450.0):
        # this conversion reproduces it from the OpenMM charmm36/water.xml k.
        return self.bond_k_kjmolnm2 * KCAL_PER_KJ / 100.0 / 2.0

    @property
    def angle_k_kcal(self) -> float:
        # Same 1/2-convention correction as bond_k_kcal_A2 (cross-checked
        # against the real CHARMM TIP3P Ktheta=55.0 kcal/mol/rad^2).
        return self.angle_k_kjmolrad2 * KCAL_PER_KJ / 2.0


MODELS = {
    "tip3p": WaterModel(
        key="tip3p",
        display_name="TIP3P (CHARMM-modified)",
        reference="Jorgensen et al., J. Chem. Phys. 1983, 79, 926",
        resname="TIP3", atom_o="OH2", type_o="OT", type_h="HT",
        mass_o=15.99940, mass_h=1.00800,
        q_o=-0.834, q_h=0.417,
        r_oh=0.9572, angle_hoh_deg=104.52,
        sigma_o_nm=0.31506, epsilon_o_kjmol=0.6364,
        sigma_h_nm=0.04000, epsilon_h_kjmol=0.1925,
        bond_k_kjmolnm2=376560.0, angle_k_kjmolrad2=460.24,
    ),
    "opc3": WaterModel(
        key="opc3",
        display_name="OPC3",
        reference="Izadi & Onufriev, J. Chem. Phys. 2016, 145, 074501",
        resname="OPC3", atom_o="O", type_o="OT3", type_h="HT3",
        mass_o=15.99940, mass_h=1.00800,
        q_o=-0.895170, q_h=0.447585,
        r_oh=0.978882, angle_hoh_deg=math.degrees(1.9106321528999624),
        sigma_o_nm=0.31742703509365927, epsilon_o_kjmol=0.683690704,
        sigma_h_nm=0.04000, epsilon_h_kjmol=0.0,
        bond_k_kjmolnm2=502416.0, angle_k_kjmolrad2=628.02,
    ),
    "tip3pfb": WaterModel(
        key="tip3pfb",
        display_name="TIP3P-FB",
        reference="Wang, Martinez & Pande, J. Phys. Chem. Lett. 2014, 5, 1885",
        resname="TP3F", atom_o="OH2", type_o="OTF", type_h="HTF",
        mass_o=15.99940, mass_h=1.00800,
        q_o=-0.848448690103, q_h=0.4242243450515,
        r_oh=1.01181082494, angle_hoh_deg=math.degrees(1.88754640288),
        sigma_o_nm=0.317796456355, epsilon_o_kjmol=0.652143528104,
        sigma_h_nm=0.04000, epsilon_h_kjmol=0.0,
        bond_k_kjmolnm2=462750.4, angle_k_kjmolrad2=836.8,
    ),
}


def get_model(key: str) -> WaterModel:
    try:
        return MODELS[key]
    except KeyError:
        raise SystemExit(f"Unknown water model '{key}'. Choose from: {', '.join(MODELS)}")


def rtf_fragment(m: WaterModel) -> str:
    return f"""\
* {m.display_name} water model topology fragment for psfgen (CHARMM/RTF format)
* {m.reference}
* Note: atoms, bonds, angle and charges only; see the companion .prm file
* for Lennard-Jones parameters.
*

MASS   -1  {m.type_o:<4s}  {m.mass_o:9.5f}  O
MASS   -1  {m.type_h:<4s}  {m.mass_h:9.5f}  H

RESI {m.resname:<4s}       0.000
GROUP
ATOM {m.atom_o:<4s} {m.type_o:<4s}  {m.q_o:9.6f}
ATOM H1   {m.type_h:<4s}  {m.q_h:9.6f}
ATOM H2   {m.type_h:<4s}  {m.q_h:9.6f}
BOND {m.atom_o} H1 {m.atom_o} H2 H1 H2
ANGLE H1 {m.atom_o} H2
PATCHING FIRS NONE LAST NONE
END
"""


def prm_fragment(m: WaterModel) -> str:
    return f"""\
* {m.display_name} water model CHARMM-format parameters
* {m.reference}
* O-H and H-H bond/angle force constants are only used for flexible-water
* runs -- with rigid water (NAMD "rigidBonds water"/SHAKE, or OpenMM
* Constraints) they are ignored and only the equilibrium length/angle
* (i.e. the geometry baked into the coordinates) matters.
*

BONDS
{m.type_o:<4s} {m.type_h:<4s}  {m.bond_k_kcal_A2:10.3f}  {m.r_oh:8.5f}
{m.type_h:<4s} {m.type_h:<4s}  {0.0:10.3f}  {m.d_hh:8.5f}

ANGLES
{m.type_h:<4s} {m.type_o:<4s} {m.type_h:<4s}  {m.angle_k_kcal:10.3f}  {m.angle_hoh_deg:8.3f}

NONBONDED
{m.type_o:<4s}   0.0  {-m.epsilon_o_kcal:10.6f}  {m.rmin2_o:8.5f}
{m.type_h:<4s}   0.0  {-m.epsilon_h_kcal:10.6f}  {m.rmin2_h:8.5f}

END
"""
