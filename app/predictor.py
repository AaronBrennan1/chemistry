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
def predict_mechanism(
    smiles: str,
    solvent: str,
    acid_base: str,
    temperature: float,
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

    steps = generate_mechanism_steps(best_mech, rearrange=rearr)
    selectivity_info = analyze_selectivity(substrate_data, condition_data)

    return format_prediction_output({
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
        "steps": steps
    })

def format_prediction_output(prediction: Dict) -> str:
    """
    Produces a human-readable summary of the mechanism prediction.
    """
    if not prediction.get("success", False):
        return f"ERROR: {prediction.get('error', 'Unknown error')}"

    result = []
    result.append("=== PREDICTED MECHANISMS (SCORES) ===")
    for mech, sc in prediction["ranking"].items():
        result.append(f"  {mech}: {sc:.1f}")

    best = prediction["best_mechanism"]
    result.append(f"\nBEST MECHANISM: {best}")
    if prediction.get("carbocation_rearrangement", False):
        result.append("(Carbocation rearrangement likely for SN1.)\n")
    else:
        result.append("")

    result.append("MECHANISM STEPS:")
    result.append(prediction["steps"] + "\n")

    sub = prediction["substrate_analysis"]
    result.append("SUBSTRATE ANALYSIS:")
    result.append(f"  Canonical SMILES: {sub['canonical_smiles']}")
    result.append(f"  Type: {sub['type']}")
    if sub['vinylic_or_aryl_halide']:
        result.append("  (Aryl or vinylic halide)")
    for feat, val in sub["features"].items():
        if val:
            result.append(f"  {feat.replace('_',' ').title()}: Yes")

    cond = prediction["condition_analysis"]
    result.append("\nCONDITION ANALYSIS:")
    result.append(f"  Solvent Type: {cond['solvent_type']}")
    result.append(f"  Environment Type: {cond['environment_type']}")
    result.append(f"  Base Strength: {cond['base_strength']}")
    result.append(f"  Nucleophile Strength: {cond['nucleophile_strength']}")
    result.append(f"  Temperature: {cond['temperature']} °C")
    result.append(f"  Pressure: {cond['pressure']} atm")

    sel = prediction["selectivity_analysis"]
    result.append("\nSELECTIVITY ANALYSIS (placeholder):")
    for k, v in sel.items():
        result.append(f"  {k}: {v}")

    return "\n".join(result)
