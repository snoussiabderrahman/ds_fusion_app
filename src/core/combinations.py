"""
Module contenant les règles de combinaison
"""
from typing import List, Dict, FrozenSet
from collections import defaultdict
from .mass import Mass
from .frame import FrameOfDiscernment
import numpy as np


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

class ZhangCombination(CombinationRule):
    """
    Règle de Zhang (méthode proposée dans le papier)
    Basée sur la transformation pignistique et la similarité cosinus
    """
    
    def __init__(self):
        super().__init__("Zhang")
        self.dempster = DempsterCombination()
    
    def pignistic_transform(self, mass: Mass) -> np.ndarray:
        """
        Transforme une masse en probabilité pignistique
        Returns: vecteur numpy de dimension n (nombre d'hypothèses)
        """
        n = mass.frame.n
        pignistic = np.zeros(n)
        
        for hypothesis, m_value in mass.get_all_masses().items():
            if m_value == 0:
                continue
            
            # Cardinalité du sous-ensemble
            card = len(hypothesis)
            
            if card > 0:
                # Distribuer équitablement sur chaque élément
                for elem in hypothesis:
                    idx = mass.frame.hypotheses.index(elem)
                    pignistic[idx] += m_value / card
        
        return pignistic
    
    def cosine_similarity(self, mass1: Mass, mass2: Mass) -> float:
        """
        Calcule la similarité cosinus entre deux masses
        via leurs transformations pignistiques
        """
        pig1 = self.pignistic_transform(mass1)
        pig2 = self.pignistic_transform(mass2)
        
        dot_product = np.dot(pig1, pig2)
        norm1 = np.linalg.norm(pig1)
        norm2 = np.linalg.norm(pig2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def build_correlation_matrix(self, masses: List[Mass]) -> np.ndarray:
        """
        Construit la matrice de corrélation S
        S[i][j] = similarité cosinus entre masses[i] et masses[j]
        """
        k = len(masses)
        S = np.ones((k, k))
        
        for i in range(k):
            for j in range(k):
                if i != j:
                    S[i][j] = self.cosine_similarity(masses[i], masses[j])
        
        return S
    
    def calculate_credibility(self, masses: List[Mass]) -> np.ndarray:
        """
        Calcule le vecteur de crédibilité Crd
        basé sur le support mutuel entre les masses
        """
        S = self.build_correlation_matrix(masses)
        k = len(masses)
        
        # Calcul du vecteur de support Sup (somme sur chaque ligne)
        Sup = np.sum(S, axis=1)
        
        # Normalisation
        Sum_Sup = np.sum(Sup)
        Crd = Sup / Sum_Sup if Sum_Sup > 0 else np.ones(k) / k
        
        return Crd
    
    def weighted_average_mass(self, masses: List[Mass], credibility: np.ndarray) -> Mass:
        """
        Calcule la moyenne pondérée des masses MAE(m)
        selon les crédibilités calculées
        """
        weighted_mass = Mass(masses[0].frame, "m_weighted")
        
        # Obtenir tous les sous-ensembles présents
        all_hypotheses = set()
        for mass in masses:
            all_hypotheses.update(mass.get_all_masses().keys())
            all_hypotheses.update(mass.get_explicit_zeros())
        
        # Calculer la moyenne pondérée
        for hyp in all_hypotheses:
            weighted_value = sum(
                credibility[i] * masses[i].get_mass(hyp)
                for i in range(len(masses))
            )
            weighted_mass.set_mass(hyp, weighted_value)
        
        return weighted_mass
    
    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Pour deux masses, utilise la méthode proposée"""
        return self.combine([m1, m2])
    
    def combine(self, masses: List[Mass]) -> Mass:
        """
        Combine plusieurs masses avec la méthode proposée:
        1. Calcul des crédibilités via similarité cosinus
        2. Moyenne pondérée des masses
        3. Application de Murphy sur la masse pondérée
        """
        if len(masses) == 0:
            raise ValueError("Au moins une masse requise")
        
        if len(masses) == 1:
            return masses[0].copy()
        
        # Étape 1: Calculer les crédibilités
        credibility = self.calculate_credibility(masses)
        
        # Étape 2: Moyenne pondérée
        weighted_mass = self.weighted_average_mass(masses, credibility)
        
        # Étape 3: Appliquer Dempster k fois sur la masse pondérée
        result = weighted_mass.copy()
        result.name = f"Zhang({','.join(m.name for m in masses)})"
        
        for _ in range(len(masses) - 1):
            result = self.dempster.combine_two(result, weighted_mass)
        
        # Préserver les zéros explicites
        self._preserve_explicit_zeros(result, masses)
        
        return result


# Registry des règles disponibles
COMBINATION_RULES = {
    "Dempster-Shafer": DempsterCombination(),
    "Smets (TBM)": SmetsCombination(),
    "Yager": YagerCombination(),
    "Murphy": MurphyCombination(),
    "PCR6": PCR6Combination(),
    "Zhang": ZhangCombination()
}


def get_rule(name: str) -> CombinationRule:
    """Récupère une règle par son nom"""
    if name not in COMBINATION_RULES:
        raise ValueError(f"Règle inconnue: {name}")
    return COMBINATION_RULES[name]