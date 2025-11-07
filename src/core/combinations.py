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
    Implémentation de la méthode proposée dans le papier (Zhang) :
    1. Calculer les crédibilités
    2. Calculer la masse pondérée
    3. Appliquer Murphy sur la masse pondérée (k fois)
    """
    
    def __init__(self):
        super().__init__("Proposed Method")
        # Vous pouvez instancier d'autres règles si nécessaire
        self.murphy = MurphyCombination()
    
    def _calculate_credibility(self, masses: List[Mass]) -> List[float]:
        """
        Calcule le vecteur de crédibilité Crd
        (Adapté de DSTheory.calculate_credibility)
        """
        # Recréer les matrices S et Sup basées sur les masses fournies
        k = len(masses)
        if k == 0: return []
        
        frame = masses[0].frame
        S = np.ones((k, k))
        
        for i in range(k):
            for j in range(k):
                if i != j:
                    # Utiliser une implémentation de cosine_similarity ici
                    # Si cosine_similarity est une méthode statique ou disponible globalement
                    pig1 = self._pignistic_transform(masses[i])
                    pig2 = self._pignistic_transform(masses[j])
                    dot_product = np.dot(pig1, pig2)
                    norm1 = np.linalg.norm(pig1)
                    norm2 = np.linalg.norm(pig2)
                    similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
                    S[i][j] = similarity
        
        Sup = np.sum(S, axis=1)
        Sum_Sup = np.sum(Sup)
        Crd = Sup / Sum_Sup if Sum_Sup > 0 else np.ones(k) / k
        
        return list(Crd) # Retourner une liste

    def _pignistic_transform(self, mass: Mass) -> np.ndarray:
        """
        Transforme une masse en probabilité pignistique
        (Adapté de DSTheory.pignistic_transform)
        """
        pignistic = np.zeros(len(mass.frame.hypotheses))
        assignments = mass.get_all_masses_with_zeros() # Inclure les zéros pour la somme
        
        for hypothesis_subset, m_value in assignments.items():
            if hypothesis_subset == mass.frame.get_theta() or m_value == 0:
                continue
                
            elements = list(hypothesis_subset)
            card = len(elements)
            
            if card > 0:
                for elem in elements:
                    try:
                        idx = mass.frame.hypotheses.index(elem)
                        pignistic[idx] += m_value / card
                    except ValueError:
                        # Element non trouvé dans le frame, ignorer
                        continue
        return pignistic

    def _weighted_average_mass(self, masses: List[Mass], credibility: List[float]) -> Mass:
        """
        Calcule la moyenne pondérée des masses MAE(m)
        (Adapté de DSTheory.weighted_average_mass)
        """
        if not masses:
            raise ValueError("Aucune masse fournie pour la moyenne pondérée")
        
        frame = masses[0].frame
        weighted_mass = Mass(frame, f"MAE({masses[0].name}...)")
        
        # Obtenir toutes les hypothèses et les zéros explicites
        all_subsets = set()
        for mass in masses:
            all_subsets.update(mass.get_all_masses().keys())
            all_subsets.update(mass.get_explicit_zeros())
        
        # Calculer la moyenne pondérée
        for subset in all_subsets:
            weighted_value = sum(
                credibility[i] * masses[i].get_mass(subset) 
                for i in range(len(masses))
            )
            weighted_mass.set_mass(subset, weighted_value)
            
        # Copier les zéros explicites si nécessaire (gestion par set_mass)
        # Il faut s'assurer que le zéro explicite est bien conservé s'il l'est dans toutes les masses d'entrée
        
        return weighted_mass

    def combine_two(self, m1: Mass, m2: Mass) -> Mass:
        """Combine deux masses en utilisant la méthode proposée"""
        # La méthode proposée combine plusieurs masses.
        # Pour combiner deux masses m1 et m2, on les traite comme une liste.
        return self.combine([m1, m2])

    def combine(self, masses: List[Mass]) -> Mass:
        """Combine plusieurs masses avec la méthode proposée"""
        if not masses:
            raise ValueError("Au moins une masse requise")
        
        # Étape 1: Calculer la crédibilité
        # Il faut s'assurer que les masses ont le même cadre
        frame = masses[0].frame
        for m in masses[1:]:
            if m.frame != frame:
                raise ValueError("Toutes les masses doivent appartenir au même cadre de discernement.")
        
        credibility = self._calculate_credibility(masses)
        
        # Étape 2: Moyenne pondérée
        weighted_mass = self._weighted_average_mass(masses, credibility)
        
        # Étape 3: Appliquer Murphy sur la masse pondérée
        # Murphy combine k masses. Ici, on applique Murphy à une seule masse répétée k fois.
        # Cela correspond à l'idée d'appliquer la combinaison sur la masse pondérée MAE(m) elle-même.
        # On peut voir cela comme une itération de Murphy sur la masse pondérée.
        result = weighted_mass.copy()
        num_original_masses = len(masses) # On utilise le nombre original pour les itérations
        for _ in range(num_original_masses - 1):
             # Utilise la combinaison de Murphy sur la masse pondérée répétée
            result = self.murphy.combine_two(result, weighted_mass)
        
        # Le résultat final est la combinaison de Murphy appliquée à la masse pondérée.
        # La méthode de Murphy interne gère déjà la combinaison.
        # Il faut s'assurer que le nom du résultat est approprié.
        result.name = f"Proposed({masses[0].name}...)" 
        
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