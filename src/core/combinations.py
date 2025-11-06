"""
Module contenant les règles de combinaison
"""
from typing import List, Dict, FrozenSet
from collections import defaultdict
from .mass import Mass
from .frame import FrameOfDiscernment


class CombinationRule:
    """Classe de base pour les règles de combinaison"""
    
    def __init__(self, name: str):
        self.name = name
    
    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses"""
        raise NotImplementedError
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Combine deux masses"""
        raise NotImplementedError
    
    def _preserve_explicit_zeros(self, result: Mass, input_masses: List[Mass]):
        """Préserve les zéros explicites des masses d'entrée dans le résultat"""
        for mass in input_masses:
            for subset in mass.get_explicit_zeros():
                if result.get_mass(subset) == 0:
                    result._explicit_zeros.add(subset)


class DempsterCombination(CombinationRule):
    """Règle de Dempster-Shafer (normalisée)"""
    
    def __init__(self):
        super().__init__("Dempster-Shafer")
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Combine deux masses avec la règle de Dempster"""
        result = Mass(m1.frame, f"{m1.name}⊕{m2.name}")
        
        # Calcul conjunctif
        assignments = defaultdict(float)
        K = 0.0  # Conflit
        
        for B, m1_val in m1.get_all_masses().items():
            for C, m2_val in m2.get_all_masses().items():
                intersection = B & C
                product = m1_val * m2_val
                
                if len(intersection) == 0:
                    K += product
                else:
                    assignments[intersection] += product
        
        # Normalisation
        if abs(K - 1.0) < 1e-10:
            raise ValueError("Conflit total (K=1), règle de Dempster non définie")
        
        denom = 1.0 - K
        for subset, value in assignments.items():
            result.set_mass(subset, value / denom)
        
        # Préserver les zéros explicites
        self._preserve_explicit_zeros(result, [m1, m2])
        
        return result
    
    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses successivement"""
        if len(masses) == 0:
            raise ValueError("Au moins une masse requise")
        
        result = masses[0].copy()
        for i in range(1, len(masses)):
            result = self.combine_two(result, masses[i])
        
        return result


class SmetsCombination(CombinationRule):
    """Règle de Smets (TBM - non normalisée)"""
    
    def __init__(self):
        super().__init__("Smets (TBM)")
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Combine deux masses avec la règle de Smets"""
        result = Mass(m1.frame, f"{m1.name}∩{m2.name}")
        
        # Calcul conjunctif sans normalisation
        assignments = defaultdict(float)
        
        for B, m1_val in m1.get_all_masses().items():
            for C, m2_val in m2.get_all_masses().items():
                intersection = B & C
                product = m1_val * m2_val
                assignments[intersection] += product
        
        for subset, value in assignments.items():
            result.set_mass(subset, value)
        
        # Préserver les zéros explicites
        self._preserve_explicit_zeros(result, [m1, m2])
        
        return result
    
    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses"""
        if len(masses) == 0:
            raise ValueError("Au moins une masse requise")
        
        result = masses[0].copy()
        for i in range(1, len(masses)):
            result = self.combine_two(result, masses[i])
        
        return result


class YagerCombination(CombinationRule):
    """Règle de Yager"""
    
    def __init__(self):
        super().__init__("Yager")
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Combine deux masses avec la règle de Yager"""
        result = Mass(m1.frame, f"{m1.name}⊕Y{m2.name}")
        
        # Calcul conjunctif
        assignments = defaultdict(float)
        K = 0.0  # Conflit
        
        for B, m1_val in m1.get_all_masses().items():
            for C, m2_val in m2.get_all_masses().items():
                intersection = B & C
                product = m1_val * m2_val
                
                if len(intersection) == 0:
                    K += product
                else:
                    assignments[intersection] += product
        
        # Le conflit est assigné à Θ
        theta = m1.frame.get_theta()
        assignments[theta] += K
        
        for subset, value in assignments.items():
            result.set_mass(subset, value)
        
        # Préserver les zéros explicites
        self._preserve_explicit_zeros(result, [m1, m2])
        
        return result
    
    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses"""
        if len(masses) == 0:
            raise ValueError("Au moins une masse requise")
        
        result = masses[0].copy()
        for i in range(1, len(masses)):
            result = self.combine_two(result, masses[i])
        
        return result


class MurphyCombination(CombinationRule):
    """Règle de Murphy (moyenne puis Dempster)"""
    
    def __init__(self):
        super().__init__("Murphy")
        self.dempster = DempsterCombination()
    
    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses avec Murphy"""
        if len(masses) == 0:
            raise ValueError("Au moins une masse requise")
        
        # Moyenne des masses
        avg_mass = Mass(masses[0].frame, "m_avg")
        
        # Collecter tous les sous-ensembles
        all_subsets = set()
        for m in masses:
            all_subsets.update(m.get_all_masses().keys())
            all_subsets.update(m.get_explicit_zeros())
        
        # Calculer la moyenne
        n = len(masses)
        for subset in all_subsets:
            avg_value = sum(m.get_mass(subset) for m in masses) / n
            avg_mass.set_mass(subset, avg_value)
        
        # Appliquer Dempster n fois sur la moyenne
        result = avg_mass.copy()
        for _ in range(n - 1):
            result = self.dempster.combine_two(result, avg_mass)
        
        # Préserver les zéros explicites
        self._preserve_explicit_zeros(result, masses)
        
        return result
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Pour deux masses, utilise Murphy"""
        return self.combine([m1, m2])


class PCR6Combination(CombinationRule):
    """Règle PCR6 (Proportional Conflict Redistribution)"""
    
    def __init__(self):
        super().__init__("PCR6")
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Combine deux masses avec PCR6"""
        result = Mass(m1.frame, f"{m1.name}⊕PCR{m2.name}")
        
        # Commencer avec la règle conjonctive
        assignments = defaultdict(float)
        conflicts = []
        
        for B, m1_val in m1.get_all_masses().items():
            for C, m2_val in m2.get_all_masses().items():
                intersection = B & C
                product = m1_val * m2_val
                
                if len(intersection) == 0:
                    conflicts.append((B, C, product))
                else:
                    assignments[intersection] += product
        
        # Redistribuer le conflit proportionnellement
        for B, C, conflict_mass in conflicts:
            m1_B = m1.get_mass(B)
            m2_C = m2.get_mass(C)
            
            if m1_B + m2_C > 0:
                # Redistribution proportionnelle
                assignments[B] += conflict_mass * m1_B / (m1_B + m2_C)
                assignments[C] += conflict_mass * m2_C / (m1_B + m2_C)
        
        for subset, value in assignments.items():
            result.set_mass(subset, value)
        
        # Préserver les zéros explicites
        self._preserve_explicit_zeros(result, [m1, m2])
        
        return result
    
    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses"""
        if len(masses) == 0:
            raise ValueError("Au moins une masse requise")
        
        result = masses[0].copy()
        for i in range(1, len(masses)):
            result = self.combine_two(result, masses[i])
        
        return result


# Registry des règles disponibles
COMBINATION_RULES = {
    "Dempster-Shafer": DempsterCombination(),
    "Smets (TBM)": SmetsCombination(),
    "Yager": YagerCombination(),
    "Murphy": MurphyCombination(),
    "PCR6": PCR6Combination()
}


def get_rule(name: str) -> CombinationRule:
    """Récupère une règle par son nom"""
    if name not in COMBINATION_RULES:
        raise ValueError(f"Règle inconnue: {name}")
    return COMBINATION_RULES[name]