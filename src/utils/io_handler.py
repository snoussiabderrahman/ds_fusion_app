"""
Module pour gérer l'import/export CSV
"""
import pandas as pd
from typing import Dict, List
from ..core.mass import Mass
from ..core.frame import FrameOfDiscernment


def export_masses_to_csv(masses: Dict[str, Mass], filename: str):
    """Exporte les masses vers un fichier CSV"""
    data = []
    
    for name, mass in masses.items():
        for subset, value in mass.get_all_masses().items():
            subset_str = mass.frame.format_subset(subset)
            data.append({
                'Mass': name,
                'Subset': subset_str,
                'Value': round(value, 4)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def export_result_to_csv(result: Mass, filename: str):
    """Exporte un résultat de fusion vers CSV"""
    data = []
    
    for subset, value in result.get_all_masses().items():
        subset_str = result.frame.format_subset(subset)
        data.append({
            'Subset': subset_str,
            'Value': round(value, 4)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def import_masses_from_csv(filename: str, frame: FrameOfDiscernment) -> Dict[str, Mass]:
    """Importe des masses depuis un fichier CSV"""
    df = pd.read_csv(filename)
    
    masses = {}
    
    for _, row in df.iterrows():
        mass_name = row['Mass']
        subset_str = row['Subset']
        value = float(row['Value'])
        
        if mass_name not in masses:
            masses[mass_name] = Mass(frame, mass_name)
        
        subset = frame.parse_subset(subset_str)
        masses[mass_name].set_mass(subset, value)
    
    return masses