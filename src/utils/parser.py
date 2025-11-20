import re
from typing import List, Tuple


def parse_mass_assignment(text: str) -> Tuple[str, str, float]:
    """
    Parse une assignation de masse: "m1(A,B) = 0.5"
    
    Returns:
        (mass_name, subset_str, value)
    """
    # Pattern: m1(A,B) = 0.5
    pattern = r'(\w+)\s*\(\s*([^)]+)\s*\)\s*=\s*([\d.]+)'
    match = re.match(pattern, text.strip())
    
    if not match:
        raise ValueError("Format invalide. Utilisez: m1(A,B) = 0.5")
    
    mass_name = match.group(1)
    subset_str = match.group(2)
    value = float(match.group(3))
    
    return mass_name, subset_str, value


def parse_fusion_expression(text: str) -> List[str]:
    """
    Parse une expression de fusion: "m1 + m2 + m3"
    
    Returns:
        Liste des noms de masses
    """
    # Séparer par +
    parts = [p.strip() for p in text.split('+')]
    
    # Valider que ce sont des noms de masses valides
    for part in parts:
        if not re.match(r'^\w+$', part):
            raise ValueError(f"Nom de masse invalide: '{part}'")
    
    return parts