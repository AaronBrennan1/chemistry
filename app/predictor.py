import sys
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union
from rdkit import Chem
from rdkit import RDLogger

# ----------------------------------------------------------------------
# 0. RDKit Logging Configuration
# ----------------------------------------------------------------------
# Redirect RDKit errors and warnings so we can capture them
# without printing them all to stdout. 
# You can adjust this if you prefer to see the raw parse logs.
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)

# ----------------------------------------------------------------------
# 1. Enumerations & Type Definitions
# ----------------------------------------------------------------------
class SubstrateType(Enum):
    METHYL = auto()
    PRIMARY = auto()
    SECONDARY = auto()
    TERTIARY = auto()
    BENZYLIC = auto()
    ALLYLIC = auto()
    NONE = auto()  # e.g., no sp3 alpha carbon with LG found

class SolventType(Enum):
    POLAR_PROTIC = auto()
    POLAR_APROTIC = auto()
    NONPOLAR = auto()
    UNKNOWN = auto()

# Simple classification: Is it an aqueous environment or organic?
class EnvironmentType(Enum):
    AQUEOUS = auto()
    ORGANIC = auto()
    UNKNOWN = auto()

class BaseStrength(Enum):
    STRONG = auto()
    WEAK = auto()
    UNKNOWN = auto()

class MechanismType(Enum):
    SN1 = "SN1"
    SN2 = "SN2"
    E1 = "E1"
    E2 = "E2"
    PERICYCLIC = "Pericyclic"
    OLEFINATION = "Olefination"  # Newly added
    SNAR = "SNAr"
    RADICAL = "Radical"

SubstrateData = Dict[str, Union[bool, str, SubstrateType]]
# ConditionData now includes temperature, pressure, environment, etc.
ConditionData = Dict[str, Union[SolventType, BaseStrength, float, EnvironmentType]]

# ----------------------------------------------------------------------
# 2. Reaction Rules / Databases
# ----------------------------------------------------------------------
REACTION_RULES = {
    MechanismType.SN2: {
        "substrate": [
            SubstrateType.METHYL,
            SubstrateType.PRIMARY,
            SubstrateType.SECONDARY,
            SubstrateType.ALLYLIC,
            SubstrateType.BENZYLIC
        ],
        "solvent_type": [SolventType.POLAR_APROTIC, SolventType.POLAR_PROTIC],
        "temperature_range": (0, 80),
    },
    MechanismType.SN1: {
        "substrate": [
            SubstrateType.SECONDARY,
            SubstrateType.TERTIARY,
            SubstrateType.BENZYLIC,
            SubstrateType.ALLYLIC
        ],
        "nucleophile_strength": BaseStrength.WEAK,
        "solvent_type": SolventType.POLAR_PROTIC,
        "temperature_range": (0, 100),
    },
    MechanismType.E2: {
        "substrate": [
            SubstrateType.SECONDARY,
            SubstrateType.TERTIARY,
            SubstrateType.ALLYLIC,
            SubstrateType.BENZYLIC
        ],
        "base_strength": BaseStrength.STRONG,
        "temperature_range": (50, 200),
    },
    MechanismType.E1: {
        "substrate": [
            SubstrateType.TERTIARY,
            SubstrateType.BENZYLIC,
            SubstrateType.ALLYLIC
        ],
        "base_strength": BaseStrength.WEAK,
        "temperature_range": (50, 200),
    },
    MechanismType.PERICYCLIC: {
        "substrate_features": ["diene", "dienophile"],
        "temperature_range": (25, 200),
    },
    # Newly added naive rule for "Olefination"
    # e.g., Wittig or related. We'll just check if there's a carbonyl present
    # and if T, P are in some naive range.
    MechanismType.OLEFINATION: {
        "requires_carbonyl": True,
        "temperature_range": (0, 150),
        "pressure_range": (1, 20),  # arbitrary (1–20 atm)
    },
    MechanismType.SNAR: {
        "vinylic_or_aryl_halide": True,
        "temperature_range": (50, 200),
    },
    MechanismType.RADICAL: {
        "base_strength": BaseStrength.STRONG,
        "solvent_type": [SolventType.NONPOLAR, SolventType.POLAR_APROTIC],
        "temperature_range": (50, 200),
    },
}

# ----------------------------------------------------------------------
# 3. Solvent & Condition Mapping
# ----------------------------------------------------------------------
SOLVENT_MAP = {
    "water": SolventType.POLAR_PROTIC,
    "ethanol": SolventType.POLAR_PROTIC,
    "methanol": SolventType.POLAR_PROTIC,
    "isopropanol": SolventType.POLAR_PROTIC,
    "t-butanol": SolventType.POLAR_PROTIC,
    "acetic acid": SolventType.POLAR_PROTIC,
    "formic acid": SolventType.POLAR_PROTIC,

    "acetone": SolventType.POLAR_APROTIC,
    "dmso": SolventType.POLAR_APROTIC,
    "dmf": SolventType.POLAR_APROTIC,
    "acetonitrile": SolventType.POLAR_APROTIC,
    "thf": SolventType.POLAR_APROTIC,

    "dichloromethane": SolventType.NONPOLAR,
    "chloroform": SolventType.NONPOLAR,
    "toluene": SolventType.NONPOLAR,
    "benzene": SolventType.NONPOLAR,
    "hexane": SolventType.NONPOLAR,
    "diethyl ether": SolventType.NONPOLAR,
    "pentane": SolventType.NONPOLAR,
}

ACID_BASE_MAP = {
    "acidic": (BaseStrength.WEAK, BaseStrength.WEAK),
    "neutral": (BaseStrength.WEAK, BaseStrength.WEAK),
    "basic": (BaseStrength.STRONG, BaseStrength.STRONG),
    "strong base": (BaseStrength.STRONG, BaseStrength.STRONG),
    "weak base": (BaseStrength.WEAK, BaseStrength.WEAK),
}

def classify_aqueous_or_organic(solvent: str) -> EnvironmentType:
    """
    Very naive. If solvent is water or mostly water-based, call it AQUEOUS.
    Otherwise, call it ORGANIC. Default = UNKNOWN if not recognized.
    """
    lower_solv = solvent.lower()
    if "water" in lower_solv:
        return EnvironmentType.AQUEOUS
    if lower_solv in SOLVENT_MAP:
        # everything else in SOLVENT_MAP we call ORGANIC for simplicity
        return EnvironmentType.ORGANIC
    return EnvironmentType.UNKNOWN

# ----------------------------------------------------------------------
# 4. SMILES Parsing & Canonicalization
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 4. SMILES Parsing & Canonicalization
# ----------------------------------------------------------------------
def sanitize_and_canonicalize_smiles(smiles: str) -> Tuple[str, Optional[Chem.Mol], bool, str]:
    """
    Attempt to parse SMILES with RDKit, then convert to canonical SMILES.
    Return (canonical_smiles, mol, success_flag, error_message).

    We append extra debugging info to help students figure out why a SMILES might fail.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            # This sometimes happens without raising an Exception but returns None
            error_hint = (
                "Could not parse SMILES. Common issues:\n"
                "  - Use 'C' instead of 'CH2' (SMILES uses atoms, not groups)\n"
                "  - Missing ring closure digits (e.g. C1...C1)\n"
                "  - Unsupported atomic symbols\n"
                "  - Disconnected fragments without '.'\n"
                "Please double-check or use a SMILES validator."
            )
            return smiles, None, False, error_hint
        cano = Chem.MolToSmiles(mol, canonical=True)
        return cano, mol, True, ""
    except Exception as e:
        # e should contain something like "SMILES Parse Error: syntax error while parsing..."
        error_hint = (
            f"SMILES parsing error from RDKit:\n{str(e)}\n\n"
            "Hints:\n"
            "  - Check for misplaced ring-closure digits (like 'C1=CC=CC=C1' for benzene)\n"
            "  - Use correct case for atoms (e.g. 'Cl' not 'CL' or 'cl')\n"
            "  - Ensure each ring digit is opened and closed exactly once\n"
            "  - If you have multiple disconnected parts, separate them with a '.'\n"
            "  - Try drawing and exporting SMILES from a tool like ChemDraw or an online converter.\n"
        )
        return smiles, None, False, error_hint

# ----------------------------------------------------------------------
# 5. Substrate Analysis Helpers
# ----------------------------------------------------------------------
def find_alpha_carbon(mol: Chem.Mol) -> Optional[Chem.Atom]:
    """
    Return the alpha carbon bonded to a halogen (F, Cl, Br, I) by a SINGLE bond, if any.
    """
    if mol is None:
        return None

    halogens: Set[str] = {"F", "Cl", "Br", "I"}

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        sym1 = a1.GetSymbol()
        sym2 = a2.GetSymbol()

        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            if (sym1 in halogens and sym2 == "C"):
                return a2
            if (sym2 in halogens and sym1 == "C"):
                return a1

    return None

def is_neopentyl_like(alpha_carbon: Chem.Atom) -> bool:
    """
    Heuristic detection for neopentyl or similarly bulky primary centers.
    """
    if alpha_carbon.GetDegree() != 2:
        return False

    # The neighbor that isn't halogen is presumably a carbon with 3 other carbons => tertiary center.
    for nbr in alpha_carbon.GetNeighbors():
        if nbr.GetSymbol() == "C":
            c_neighbors = sum(1 for x in nbr.GetNeighbors() if x.GetSymbol() == "C")
            if c_neighbors >= 3:
                return True
    return False

def detect_carbocation_rearrangement_possible(alpha_carbon: Chem.Atom) -> bool:
    """
    Naive check: if alpha carbon is secondary, and there's an adjacent tertiary carbon,
    guess that a rearrangement could be favorable.
    """
    if alpha_carbon.GetDegree() == 3:
        for nbr in alpha_carbon.GetNeighbors():
            if nbr.GetSymbol() == "C":
                c_neighbors = sum(1 for x in nbr.GetNeighbors() if x.GetSymbol() == "C")
                if c_neighbors >= 3:
                    return True
    return False

def contains_carbonyl(mol: Chem.Mol) -> bool:
    """Check if the molecule contains a C=O double bond."""
    if mol is None:
        return False
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            # Carbonyl if one is C and the other is O
            if {a1.GetSymbol(), a2.GetSymbol()} == {"C", "O"}:
                return True
    return False

# ----------------------------------------------------------------------
# 6. Main Substrate Classification
# ----------------------------------------------------------------------
def classify_substrate(smiles: str) -> Dict:
    """
    Classify the substrate: sp3 alpha carbon, allylic/benzylic, vinylic/aryl, neopentyl, etc.
    Also check for diene/dienophile, carbonyl (for naive "olefination"), etc.
    """
    canonical_smiles, mol, success, error = sanitize_and_canonicalize_smiles(smiles)
    if not success or (mol is None):
        return {"valid": False, "reason": error}

    alpha_carbon = find_alpha_carbon(mol)
    has_leaving_group = (alpha_carbon is not None)

    # Count double bonds, check aromatic
    has_double_bond = any(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in mol.GetBonds())
    has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    double_bond_count = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.DOUBLE)

    is_diene = (double_bond_count >= 2) and not has_aromatic
    is_dienophile = (double_bond_count == 1) and not has_aromatic
    has_carbonyl = contains_carbonyl(mol)

    # If no alpha carbon => might be pericyclic or something else
    if alpha_carbon is None:
        return {
            "valid": True,
            "canonical_smiles": canonical_smiles,
            "substrate_type": SubstrateType.NONE,
            "vinylic_or_aryl_halide": False,
            "has_double_bond": has_double_bond,
            "has_aromatic": has_aromatic,
            "is_diene": is_diene,
            "is_dienophile": is_dienophile,
            "has_leaving_group": False,
            "is_neopentyl": False,
            "carbocation_rearrangement_possible": False,
            "has_carbonyl": has_carbonyl,
        }

    # Check if alpha carbon is sp2 or aromatic => vinylic or aryl halide
    alpha_carbon_hyb = alpha_carbon.GetHybridization()
    is_vinylic_or_aryl = (
        alpha_carbon_hyb == Chem.rdchem.HybridizationType.SP2
        or alpha_carbon.GetIsAromatic()
    )

    if is_vinylic_or_aryl:
        # Vinylic or aryl halide
        return {
            "valid": True,
            "canonical_smiles": canonical_smiles,
            "substrate_type": SubstrateType.NONE,  # not a typical sp3 center
            "vinylic_or_aryl_halide": True,
            "has_double_bond": has_double_bond,
            "has_aromatic": has_aromatic,
            "is_diene": is_diene,
            "is_dienophile": is_dienophile,
            "has_leaving_group": True,
            "is_neopentyl": False,
            "carbocation_rearrangement_possible": False,
            "has_carbonyl": has_carbonyl,
        }

    # If sp3:
    connected_carbons = sum(1 for nbr in alpha_carbon.GetNeighbors() if nbr.GetSymbol() == "C")
    if connected_carbons == 0:
        base_type = SubstrateType.METHYL
    elif connected_carbons == 1:
        base_type = SubstrateType.PRIMARY
    elif connected_carbons == 2:
        base_type = SubstrateType.SECONDARY
    elif connected_carbons == 3:
        base_type = SubstrateType.TERTIARY
    else:
        return {"valid": False, "reason": "Unusual carbon connectivity."}

    final_type = base_type

    # Check benzylic or allylic
    for nbr in alpha_carbon.GetNeighbors():
        if nbr.GetIsAromatic():
            final_type = SubstrateType.BENZYLIC
            break
        # If neighbor is part of a double bond => allylic
        for bnd in nbr.GetBonds():
            if bnd.GetBondType() == Chem.rdchem.BondType.DOUBLE and bnd.GetOtherAtom(nbr) != alpha_carbon:
                final_type = SubstrateType.ALLYLIC
                break

    # Check neopentyl-like
    neo_flag = False
    if final_type == SubstrateType.PRIMARY:
        if is_neopentyl_like(alpha_carbon):
            neo_flag = True

    # Check for possible carbocation rearrangement
    cation_rearrange_possible = False
    if final_type == SubstrateType.SECONDARY:
        if detect_carbocation_rearrangement_possible(alpha_carbon):
            cation_rearrange_possible = True

    return {
        "valid": True,
        "canonical_smiles": canonical_smiles,
        "substrate_type": final_type,
        "vinylic_or_aryl_halide": False,
        "has_double_bond": has_double_bond,
        "has_aromatic": has_aromatic,
        "is_diene": is_diene,
        "is_dienophile": is_dienophile,
        "has_leaving_group": True,
        "is_neopentyl": neo_flag,
        "carbocation_rearrangement_possible": cation_rearrange_possible,
        "has_carbonyl": has_carbonyl,
    }

# ----------------------------------------------------------------------
# 7. Functions to Evaluate Conditions
# ----------------------------------------------------------------------
def classify_conditions(
    solvent: str,
    acid_base: str,
    temperature: float,
    pressure: float = 1.0
) -> ConditionData:
    """
    Build a dictionary describing the reaction conditions:
    solvent type (polar protic, aprotic, etc.), environment (aqueous vs. organic),
    base strength, nucleophile strength, temperature, and pressure.
    """
    solvent_type = SOLVENT_MAP.get(solvent.lower(), SolventType.UNKNOWN)
    base_strength, nucleophile_strength = ACID_BASE_MAP.get(
        acid_base.lower(), (BaseStrength.UNKNOWN, BaseStrength.UNKNOWN)
    )
    env_type = classify_aqueous_or_organic(solvent)

    return {
        "solvent_type": solvent_type,
        "environment_type": env_type,
        "base_strength": base_strength,
        "nucleophile_strength": nucleophile_strength,
        "temperature": temperature,
        "pressure": pressure,
    }

# ----------------------------------------------------------------------
# 8. Selectivity Analysis (Placeholder)
# ----------------------------------------------------------------------
def analyze_selectivity(substrate_data: Dict, condition_data: Dict) -> Dict[str, str]:
    """
    Placeholder for deeper analysis of selectivity (regio-, chemo-, stereo-...).
    Extend or replace with real logic as needed.
    """
    # Example placeholders:
    return {
        "regioselectivity": "Likely determined by most substituted site",
        "stereoselectivity": "Depends on reaction specifics (concerted vs. stepwise).",
        "chemoselectivity": "No competing functional groups detected." 
    }

# ----------------------------------------------------------------------
# 9. Scoring/Ranking Mechanisms
# ----------------------------------------------------------------------
def rank_mechanisms(substrate_data: Dict, condition_data: Dict) -> Dict[str, float]:
    """
    Returns a dictionary of {mechanism_name: score}, where score is 0 to 100.
    A score of 0 means "not applicable." Higher scores mean more favored.
    """
    scores = {m.value: 0.0 for m in MechanismType}

    if not substrate_data.get("valid", False):
        return scores  # all zero

    # Quick handle PERICYCLIC (if no typical LG or even if there is, but conditions match)
    if substrate_data["is_diene"] or substrate_data["is_dienophile"]:
        rule = REACTION_RULES[MechanismType.PERICYCLIC]
        min_t, max_t = rule["temperature_range"]
        t = condition_data["temperature"]
        if min_t <= t <= max_t and not substrate_data["has_aromatic"]:
            scores[MechanismType.PERICYCLIC.value] = 80.0

    # Check for Olefination (naive: if there's a carbonyl and T,P in range)
    if substrate_data["has_carbonyl"]:
        olef_rule = REACTION_RULES[MechanismType.OLEFINATION]
        min_t, max_t = olef_rule["temperature_range"]
        min_p, max_p = olef_rule["pressure_range"]
        t = condition_data["temperature"]
        p = condition_data["pressure"]
        if min_t <= t <= max_t and min_p <= p <= max_p:
            scores[MechanismType.OLEFINATION.value] = 70.0  # some base value

    # If there's a leaving group (sp3 or sp2)
    if substrate_data["has_leaving_group"]:
        # If vinylic/aryl => normal SN2/SN1/E2/E1 are mostly blocked
        # but SNAr is possible
        if substrate_data["vinylic_or_aryl_halide"]:
            snar_rule = REACTION_RULES[MechanismType.SNAR]
            min_t, max_t = snar_rule["temperature_range"]
            t = condition_data["temperature"]
            if min_t <= t <= max_t:
                scores[MechanismType.SNAR.value] = 80.0
        else:
            # Normal sp3 alpha carbon
            stype = substrate_data["substrate_type"]
            base = condition_data["base_strength"]
            nuc_str = condition_data["nucleophile_strength"]
            solv = condition_data["solvent_type"]
            temp = condition_data["temperature"]

            # SN2
            if not substrate_data["is_neopentyl"]:
                sn2_rule = REACTION_RULES[MechanismType.SN2]
                if stype in sn2_rule["substrate"]:
                    if solv in sn2_rule["solvent_type"]:
                        tmin, tmax = sn2_rule["temperature_range"]
                        if tmin <= temp <= tmax:
                            scores[MechanismType.SN2.value] = 50.0
                            # Methyl or primary => boost SN2
                            if stype in [SubstrateType.METHYL, SubstrateType.PRIMARY]:
                                scores[MechanismType.SN2.value] += 30.0
                            # If strong base + secondary => partial credit
                            if stype == SubstrateType.SECONDARY and base == BaseStrength.STRONG:
                                scores[MechanismType.SN2.value] += 10.0

            # SN1
            sn1_rule = REACTION_RULES[MechanismType.SN1]
            if stype in sn1_rule["substrate"]:
                if solv == sn1_rule["solvent_type"]:
                    tmin, tmax = sn1_rule["temperature_range"]
                    if tmin <= temp <= tmax:
                        if nuc_str == sn1_rule["nucleophile_strength"]:
                            scores[MechanismType.SN1.value] = 50.0
                            if stype == SubstrateType.TERTIARY:
                                scores[MechanismType.SN1.value] += 30.0
                            if stype in [SubstrateType.ALLYLIC, SubstrateType.BENZYLIC]:
                                scores[MechanismType.SN1.value] += 20.0

            # E2
            e2_rule = REACTION_RULES[MechanismType.E2]
            if stype in e2_rule["substrate"]:
                if base == e2_rule["base_strength"]:
                    tmin, tmax = e2_rule["temperature_range"]
                    if tmin <= temp <= tmax:
                        scores[MechanismType.E2.value] = 50.0
                        if stype == SubstrateType.TERTIARY:
                            scores[MechanismType.E2.value] += 25.0
                        if temp > 80:
                            scores[MechanismType.E2.value] += 15.0

            # E1
            e1_rule = REACTION_RULES[MechanismType.E1]
            if stype in e1_rule["substrate"]:
                if base == e1_rule["base_strength"]:
                    tmin, tmax = e1_rule["temperature_range"]
                    if tmin <= temp <= tmax:
                        scores[MechanismType.E1.value] = 40.0
                        if stype == SubstrateType.TERTIARY:
                            scores[MechanismType.E1.value] += 30.0
                        if temp > 80:
                            scores[MechanismType.E1.value] += 10.0

    # RADICAL
    rad_rule = REACTION_RULES[MechanismType.RADICAL]
    if (condition_data["base_strength"] == rad_rule["base_strength"] and
        condition_data["solvent_type"] in rad_rule["solvent_type"]):
        tmin, tmax = rad_rule["temperature_range"]
        if tmin <= condition_data["temperature"] <= tmax:
            scores[MechanismType.RADICAL.value] = 40.0

    # Clamp final scores
    for key in scores:
        if scores[key] < 0:
            scores[key] = 0
        if scores[key] > 100:
            scores[key] = 100

    return scores

# ----------------------------------------------------------------------
# 10. Mechanism Steps & Summaries
# ----------------------------------------------------------------------
def generate_mechanism_steps(mechanism: str, rearrange: bool=False) -> str:
    """
    If rearrange=True and mechanism is SN1, mention a possible carbocation rearrangement step.
    """
    if mechanism == "SN2":
        return (
            "1) Nu⁻ + R–LG → [Nu…R…LG]‡ → R–Nu + LG⁻\n"
            "   (Backside attack, concerted displacement)"
        )
    elif mechanism == "SN1":
        steps = (
            "1) R–LG → R⁺ + LG⁻  (rate-determining ionization)\n"
            "2) R⁺ + Nu → R–Nu   (nucleophile attacks carbocation)"
        )
        if rearrange:
            steps += "\n*Note:* Possible carbocation rearrangement before nucleophilic attack."
        return steps
    elif mechanism == "E2":
        return (
            "1) B⁻ + R–CH–LG → [TS] → R=CH + B–H + LG⁻\n"
            "   (Concerted, anti-periplanar elimination)"
        )
    elif mechanism == "E1":
        return (
            "1) R–LG → R⁺ + LG⁻  (rate-determining ionization)\n"
            "2) R⁺ + B → R=CH + B–H  (deprotonation of carbocation)"
        )
    elif mechanism == "Pericyclic":
        return (
            "Concerted π-system reorganization (e.g., Diels–Alder):\n"
            "  diene + dienophile → cycloadduct"
        )
    elif mechanism == "Olefination":
        return (
            "Typical olefination (e.g. Wittig):\n"
            "1) Generate ylide (if applicable)\n"
            "2) Ylide attacks carbonyl, forming betaine/oxaphosphetane\n"
            "3) Collapse to form alkene + by-product"
        )
    elif mechanism == "SNAr":
        return (
            "Aromatic nucleophilic substitution (addition-elimination) on an activated aryl halide:\n"
            "1) Nu attacks ipso carbon → Meisenheimer intermediate\n"
            "2) Elimination of LG reforms aromatic ring"
        )
    elif mechanism == "Radical":
        return (
            "Radical mechanism (simplified):\n"
            "1) Initiation (homolytic cleavage → radicals)\n"
            "2) Propagation (chain reaction)\n"
            "3) Termination (radicals combine)"
        )
    else:
        return f"No steps available for '{mechanism}'"

# ----------------------------------------------------------------------
# 11. Higher-Level Prediction Functions
# ----------------------------------------------------------------------
def generate_detailed_mechanism_steps(mechanism: str, substrate_data: Dict, condition_data: Dict, rearrange: bool=False) -> str:
    """
    Generates a detailed, substrate-specific mechanism description based on the 
    substrate structure and reaction conditions.
    
    Args:
        mechanism: The type of mechanism (SN1, SN2, etc.)
        substrate_data: Dictionary containing substrate analysis
        condition_data: Dictionary containing condition analysis
        rearrange: Boolean indicating if carbocation rearrangement is likely
        
    Returns:
        A string containing detailed mechanism steps specific to the substrate
    """
    # Extract useful information for description
    substrate_type = substrate_data["substrate_type"].name if isinstance(substrate_data["substrate_type"], Enum) else substrate_data["substrate_type"]
    canonical_smiles = substrate_data.get("canonical_smiles", "")
    
    # Get solvent type and other condition details
    solvent_type = condition_data["solvent_type"].name if isinstance(condition_data["solvent_type"], Enum) else condition_data["solvent_type"]
    is_benzylic = substrate_type == "BENZYLIC"
    is_allylic = substrate_type == "ALLYLIC"
    
    # Building blocks for mechanism descriptions
    substrate_desc = f"{substrate_type.lower()} substrate"
    if is_benzylic:
        substrate_desc = "benzylic substrate (stabilized by resonance with aromatic ring)"
    elif is_allylic:
        substrate_desc = "allylic substrate (stabilized by resonance with π-bond)"
    
    # Detailed mechanism steps based on mechanism type
    if mechanism == "SN2":
        description = (
            f"SN2 Mechanism with {substrate_desc} ({canonical_smiles}):\n\n"
            f"1) The nucleophile approaches the alpha carbon from the backside, opposite to the leaving group\n"
            f"   - In {solvent_type} solvent, the nucleophile's reactivity is {'enhanced' if solvent_type == 'POLAR_APROTIC' else 'somewhat diminished'}\n"
            f"2) A concerted process occurs where the nucleophile attacks while the leaving group departs\n"
            f"   - The alpha carbon undergoes inversion of configuration as the nucleophile attacks\n"
            f"   - A pentacoordinate transition state forms momentarily [Nu...C...LG]‡\n"
            f"3) The C-LG bond breaks as the Nu-C bond forms, yielding product with inverted stereochemistry\n"
            f"   - The reaction proceeds in a single step with second-order kinetics\n"
            f"   - Rate = k[substrate][nucleophile]"
        )
        
        # Add specific details based on substrate type
        if substrate_type == "METHYL":
            description += "\n\n*Note:* With a methyl substrate, steric hindrance is minimal, making SN2 highly favorable."
        elif substrate_type == "PRIMARY":
            description += "\n\n*Note:* Primary substrates have low steric hindrance around the reaction center, facilitating backside attack."
        elif substrate_type == "SECONDARY":
            description += "\n\n*Note:* Some steric hindrance is present, but the backside attack is still accessible."
        elif is_benzylic or is_allylic:
            description += f"\n\n*Note:* The {substrate_type.lower()[:-2]}ic position allows SN2 to compete with SN1 due to resonance stabilization of the transition state."
        
        return description
        
    elif mechanism == "SN1":
        description = (
            f"SN1 Mechanism with {substrate_desc} ({canonical_smiles}):\n\n"
            f"1) Heterolytic cleavage of the C-LG bond (rate-determining step)\n"
            f"   - The leaving group departs with the electron pair, forming a carbocation intermediate\n"
            f"   - {solvent_type} solvent {'helps stabilize the charged intermediates' if solvent_type == 'POLAR_PROTIC' else 'is not optimal for stabilizing the charged intermediates'}\n"
            f"2) The carbocation intermediate forms, with sp² hybridization (planar geometry)"
        )
        
        if is_benzylic:
            description += "\n   - The positive charge is delocalized through resonance with the aromatic ring, stabilizing the carbocation"
        elif is_allylic:
            description += "\n   - The positive charge is delocalized through resonance with the adjacent π-bond, stabilizing the carbocation"
        elif substrate_type == "TERTIARY":
            description += "\n   - The carbocation is stabilized by the electron-donating inductive effect of the three alkyl groups"
        
        if rearrange:
            description += (
                f"\n3) Carbocation rearrangement occurs to form a more stable carbocation\n"
                f"   - Likely either a hydride shift (1,2-H shift) or alkyl shift to form a more substituted carbocation"
            )
            next_step = 4
        else:
            next_step = 3
        
        description += (
            f"\n{next_step}) Nucleophilic attack on the (possibly rearranged) carbocation\n"
            f"   - The nucleophile can approach from either face of the planar carbocation\n"
            f"   - This typically results in racemization if the carbon was a stereocenter\n"
            f"{next_step+1}) Final product formation with first-order kinetics\n"
            f"   - Rate = k[substrate], independent of nucleophile concentration"
        )
        
        return description
        
    elif mechanism == "E2":
        description = (
            f"E2 Mechanism with {substrate_desc} ({canonical_smiles}):\n\n"
            f"1) The base approaches the β-hydrogen (a hydrogen on a carbon adjacent to the one with the leaving group)\n"
            f"2) Concerted process with anti-periplanar geometry:\n"
            f"   - The base abstracts the β-hydrogen as a proton\n"
            f"   - The electrons from the C-H bond move to form a π-bond between the α and β carbons\n"
            f"   - The electrons from the C-LG bond move to the leaving group\n"
            f"3) Formation of an alkene product and the leaving group anion\n"
            f"   - The reaction proceeds with second-order kinetics\n"
            f"   - Rate = k[substrate][base]"
        )
        
        # Add specific details based on substrate type
        temperature = condition_data.get("temperature", 25.0)
        if substrate_type == "TERTIARY" or temperature > 80:
            description += (
                f"\n\n*Note:* {'Tertiary substrates favor E2 over SN2 due to steric hindrance around the alpha carbon' if substrate_type == 'TERTIARY' else ''}"
                f"{' and' if substrate_type == 'TERTIARY' and temperature > 80 else ''}"
                f"{' Higher temperature ('+str(temperature)+' °C) increases the preference for elimination over substitution' if temperature > 80 else ''}"
            )
        
        # Add details about possible Zaitsev vs Hofmann products
        if substrate_type in ["SECONDARY", "TERTIARY"]:
            description += "\n\n*Regioselectivity:* If multiple β-hydrogens are available, the reaction typically follows Zaitsev's rule, forming the more substituted alkene (thermodynamic product)."
        
        return description
        
    elif mechanism == "E1":
        description = (
            f"E1 Mechanism with {substrate_desc} ({canonical_smiles}):\n\n"
            f"1) Heterolytic cleavage of the C-LG bond (rate-determining step)\n"
            f"   - The leaving group departs with the electron pair, forming a carbocation intermediate\n"
            f"   - {solvent_type} solvent {'helps stabilize the charged intermediates' if solvent_type == 'POLAR_PROTIC' else 'is not optimal for stabilizing the charged intermediates'}\n"
            f"2) The carbocation intermediate forms, with sp² hybridization (planar geometry)"
        )
        
        if is_benzylic:
            description += "\n   - The positive charge is delocalized through resonance with the aromatic ring, stabilizing the carbocation"
        elif is_allylic:
            description += "\n   - The positive charge is delocalized through resonance with the adjacent π-bond, stabilizing the carbocation"
        elif substrate_type == "TERTIARY":
            description += "\n   - The carbocation is stabilized by the electron-donating inductive effect of the three alkyl groups"
        
        if rearrange:
            description += (
                f"\n3) Carbocation rearrangement occurs to form a more stable carbocation\n"
                f"   - Likely either a hydride shift (1,2-H shift) or alkyl shift to form a more substituted carbocation"
            )
            next_step = 4
        else:
            next_step = 3
        
        description += (
            f"\n{next_step}) Base-mediated deprotonation of a β-hydrogen\n"
            f"   - The weak base removes a proton from a carbon adjacent to the carbocation\n"
            f"   - The electrons from the C-H bond move to form a π-bond between the α and β carbons\n"
            f"{next_step+1}) Formation of an alkene product\n"
            f"   - The reaction proceeds with first-order kinetics\n"
            f"   - Rate = k[substrate], independent of base concentration\n"
            f"   - Typically follows Zaitsev's rule (more substituted alkene is favored)"
        )
        
        return description
        
    elif mechanism == "Pericyclic":
        if substrate_data.get("is_diene", False) and substrate_data.get("is_dienophile", False):
            return "Error: Both diene and dienophile can't be in the same molecule for intermolecular Diels-Alder."
        
        if substrate_data.get("is_diene", False):
            description = (
                f"Pericyclic Mechanism (Diels-Alder) with diene ({canonical_smiles}):\n\n"
                f"1) The diene adopts an s-cis conformation (if acyclic)\n"
                f"2) Concerted cycloaddition occurs with a dienophile:\n"
                f"   - The π electrons flow in a cyclic fashion\n"
                f"   - Three π bonds are broken, and two new σ bonds and one new π bond form\n"
                f"3) Formation of a cyclohexene derivative via [4+2] cycloaddition\n"
                f"   - The reaction is stereospecific (preserves stereochemistry of reactants)\n"
                f"   - Typically follows endo selectivity due to secondary orbital interactions"
            )
        elif substrate_data.get("is_dienophile", False):
            description = (
                f"Pericyclic Mechanism (Diels-Alder) with dienophile ({canonical_smiles}):\n\n"
                f"1) Dienophile approaches a diene (which should be in s-cis conformation if acyclic)\n"
                f"2) Concerted cycloaddition occurs:\n"
                f"   - The π electrons flow in a cyclic fashion\n"
                f"   - Three π bonds are broken, and two new σ bonds and one new π bond form\n"
                f"3) Formation of a cyclohexene derivative via [4+2] cycloaddition\n"
                f"   - The reaction is stereospecific (preserves stereochemistry of reactants)\n"
                f"   - Electron-withdrawing groups on the dienophile enhance reactivity"
            )
        else:
            description = (
                f"Pericyclic Mechanism with ({canonical_smiles}):\n\n"
                f"Generic pericyclic process with concerted electron reorganization\n"
                f"This could potentially involve:\n"
                f"- Electrocyclic ring opening/closing\n"
                f"- Sigmatropic rearrangement\n"
                f"- Group transfer reaction\n"
                f"Details depend on specific substrate features not fully analyzed"
            )
        
        return description
        
    elif mechanism == "Olefination":
        if substrate_data.get("has_carbonyl", False):
            description = (
                f"Olefination Mechanism with carbonyl compound ({canonical_smiles}):\n\n"
                f"1) A phosphorus ylide (typically Wittig reagent, Ph₃P=CR₂) approaches the carbonyl carbon\n"
                f"2) Nucleophilic attack of the ylide on the carbonyl carbon:\n"
                f"   - The ylide's carbanion attacks the electrophilic carbonyl carbon\n"
                f"   - Forms a four-membered oxaphosphetane intermediate\n"
                f"3) Decomposition of the oxaphosphetane:\n"
                f"   - The four-membered ring collapses\n"
                f"   - Formation of a C=C bond and triphenylphosphine oxide (Ph₃P=O)\n"
                f"4) Formation of an alkene product:\n"
                f"   - The geometry of the alkene (E/Z) depends on reaction conditions and ylide structure\n"
                f"   - Stabilized ylides tend to give E-alkenes (thermodynamic control)\n"
                f"   - Non-stabilized ylides tend to give Z-alkenes (kinetic control)"
            )
        else:
            description = (
                f"Olefination Mechanism would require a carbonyl compound, which appears to be missing in ({canonical_smiles}):\n\n"
                f"This substrate lacks a carbonyl group for typical olefination reactions like Wittig."
            )
        
        return description
        
    elif mechanism == "SNAr":
        if substrate_data.get("vinylic_or_aryl_halide", False) and substrate_data.get("has_aromatic", False):
            description = (
                f"SNAr Mechanism with aryl halide ({canonical_smiles}):\n\n"
                f"1) Nucleophilic attack at the ipso carbon (carbon bonded to the leaving group):\n"
                f"   - The nucleophile attacks the electrophilic carbon bearing the leaving group\n"
                f"   - Formation of a Meisenheimer complex (anionic σ-adduct)\n"
                f"   - The negative charge is delocalized through resonance in the aromatic ring\n"
                f"2) Elimination of the leaving group:\n"
                f"   - Departure of the leaving group\n"
                f"   - Rearomatization of the ring\n"
                f"3) Formation of the substitution product\n"
                f"   - Addition-elimination mechanism (not concerted)"
            )
            
            # Add note about activating groups if present
            description += (
                f"\n\n*Note:* SNAr reactions are facilitated by electron-withdrawing groups (EWG) in the ortho/para positions\n"
                f"which stabilize the negative charge in the Meisenheimer complex."
            )
        else:
            description = (
                f"SNAr Mechanism with ({canonical_smiles}):\n\n"
                f"SNAr typically requires an aryl halide with electron-withdrawing groups.\n"
                f"The current substrate may not be ideal for SNAr."
            )
        
        return description
        
    elif mechanism == "Radical":
        description = (
            f"Radical Mechanism with ({canonical_smiles}):\n\n"
            f"1) Initiation:\n"
            f"   - Homolytic cleavage of an initiator (e.g., peroxide) to form radicals\n"
            f"   - The radical abstracts an atom (typically hydrogen or halogen) from the substrate\n"
            f"2) Propagation:\n"
            f"   - The resulting substrate radical undergoes further reactions\n"
            f"   - This may include addition to π bonds, fragmentation, or rearrangement\n"
            f"   - Each step generates a new radical that continues the chain reaction\n"
            f"3) Termination:\n"
            f"   - Two radicals combine to form a stable product\n"
            f"   - This step ends the chain reaction"
        )
        
        if is_benzylic or is_allylic:
            description += (
                f"\n\n*Note:* {substrate_type.lower()} positions are particularly susceptible to radical reactions\n"
                f"due to resonance stabilization of the resulting radical intermediate."
            )
        
        return description
    
    else:
        return f"No detailed steps available for '{mechanism}' with this substrate"

def predict_mechanism(
    smiles: str,
    solvent: str,
    acid_base: str,
    temperature: float = 25.0,
    pressure: float = 1.0
) -> Dict:
    """
    Returns a dictionary with:
      {
        "success": bool,
        "ranking": { mechanism_name: score },
        "best_mechanism": str,
        "carbocation_rearrangement": bool,
        "substrate_analysis": {...},
        "condition_analysis": {...},
        "selectivity_analysis": {...},
        "steps": "...",
        "detailed_steps": "...",  # New field with substrate-specific mechanism
        "error": optional error message
      }
    """
    substrate_data = classify_substrate(smiles)
    if not substrate_data.get("valid", False):
        return {"success": False, "error": substrate_data.get("reason", "Invalid substrate")}

    condition_data = classify_conditions(solvent, acid_base, temperature, pressure)
    scores = rank_mechanisms(substrate_data, condition_data)

    if all(s == 0 for s in scores.values()):
        return {
            "success": False,
            "ranking": scores,
            "error": "No mechanism matched (all scores = 0)."
        }

    sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_mech, best_score = sorted_by_score[0]

    # Flag for carbocation rearrangement
    rearr = (best_mech == MechanismType.SN1.value) and substrate_data.get("carbocation_rearrangement_possible", False)

    # Generate both generic and detailed mechanism steps
    generic_steps = generate_mechanism_steps(best_mech, rearrange=rearr)
    detailed_steps = generate_detailed_mechanism_steps(best_mech, substrate_data, condition_data, rearrange=rearr)
    
    selectivity_info = analyze_selectivity(substrate_data, condition_data)

    return {
        "success": True,
        "ranking": dict(sorted_by_score),
        "best_mechanism": best_mech,
        "carbocation_rearrangement": rearr,
        "substrate_analysis": {
            "canonical_smiles": substrate_data.get("canonical_smiles", ""),
            "type": substrate_data["substrate_type"].name,
            "vinylic_or_aryl_halide": substrate_data["vinylic_or_aryl_halide"],
            "features": {
                "has_double_bond": substrate_data["has_double_bond"],
                "has_aromatic": substrate_data["has_aromatic"],
                "is_diene": substrate_data["is_diene"],
                "is_dienophile": substrate_data["is_dienophile"],
                "is_neopentyl": substrate_data["is_neopentyl"],
                "has_carbonyl": substrate_data["has_carbonyl"],
            }
        },
        "condition_analysis": {
            "solvent_type": condition_data["solvent_type"].name,
            "environment_type": condition_data["environment_type"].name,
            "base_strength": condition_data["base_strength"].name,
            "nucleophile_strength": condition_data["nucleophile_strength"].name,
            "temperature": condition_data["temperature"],
            "pressure": condition_data["pressure"],
        },
        "selectivity_analysis": selectivity_info,
        "steps": generic_steps,
        "detailed_steps": detailed_steps
    }

def format_prediction_output(prediction: Dict) -> str:
    """
    Produces an HTML-formatted summary of the mechanism prediction with detailed steps.
    """
    # If there's an error, just return a simple HTML paragraph.
    if not prediction.get("success", False):
        return f"<p><strong>ERROR:</strong> {prediction.get('error', 'Unknown error')}</p>"

    # Start building an array of HTML segments, which we'll join at the end.
    html_output = []
    
    # Header for the mechanism scores
    html_output.append("<h3>Predicted Mechanisms (Scores)</h3>")
    html_output.append("<ul>")
    for mech, score in prediction["ranking"].items():
        html_output.append(f"<li>{mech}: {score:.1f}</li>")
    html_output.append("</ul>")

    # Best mechanism, plus carbocation rearrangement message if applicable
    best = prediction["best_mechanism"]
    html_output.append(f"<h4>Best Mechanism: {best}</h4>")
    if prediction.get("carbocation_rearrangement", False):
        html_output.append("<p><strong>Carbocation rearrangement likely for SN1.</strong></p>")
    
    # Detailed mechanism: replace newline characters with <br>, and replace '*Note:*' with a bold "Note:".
    detailed_steps = prediction.get("detailed_steps", "Not available")
    # Convert newlines to <br> for HTML
    detailed_steps_html = detailed_steps.replace("\n", "<br>")
    # Replace '*Note:*' with HTML bold, for example
    detailed_steps_html = detailed_steps_html.replace("*Note:*", "<strong>Note:</strong>")

    html_output.append("<h4>Detailed Mechanism</h4>")
    html_output.append(f"<p>{detailed_steps_html}</p>")

    # Substrate analysis
    sub = prediction["substrate_analysis"]
    html_output.append("<h4>Substrate Analysis</h4>")
    html_output.append(f"<p>Canonical SMILES: {sub['canonical_smiles']}<br>")
    html_output.append(f"Type: {sub['type']}<br>")
    if sub['vinylic_or_aryl_halide']:
        html_output.append("(Aryl or vinylic halide)<br>")
    
    # Any 'features' that are True, list them
    feature_lines = []
    for feat, val in sub["features"].items():
        if val:
            nice_feat_name = feat.replace("_", " ").title()  # e.g. "has_aromatic" -> "Has Aromatic"
            feature_lines.append(nice_feat_name)
    if feature_lines:
        html_output.append("Features:<br> • " + "<br> • ".join(feature_lines))

    html_output.append("</p>")

    # Condition analysis
    cond = prediction["condition_analysis"]
    html_output.append("<h4>Condition Analysis</h4>")
    html_output.append(f"<p>Solvent Type: {cond['solvent_type']}<br>")
    html_output.append(f"Environment Type: {cond['environment_type']}<br>")
    html_output.append(f"Base Strength: {cond['base_strength']}<br>")
    html_output.append(f"Nucleophile Strength: {cond['nucleophile_strength']}<br>")
    html_output.append(f"Temperature: {cond['temperature']} °C<br>")
    html_output.append(f"Pressure: {cond['pressure']} atm</p>")

    # Selectivity analysis
    sel = prediction["selectivity_analysis"]
    html_output.append("<h4>Selectivity Analysis</h4>")
    html_output.append("<ul>")
    for key, val in sel.items():
        html_output.append(f"<li>{key}: {val}</li>")
    html_output.append("</ul>")

    # Join everything into a single string of HTML
    return "\n".join(html_output)
