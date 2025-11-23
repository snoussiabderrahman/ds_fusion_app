from typing import List, Dict, FrozenSet
from collections import defaultdict
from .mass import Mass
from .frame import FrameOfDiscernment
import numpy as np

def get_ordered_powerset(frame: 'FrameOfDiscernment') -> List[FrozenSet]:
    """Retourne le power set dans un ordre déterministe (vital pour Jousselme)"""
    # Suppose que frame.get_power_set() retourne une liste de frozensets
    # On trie par taille, puis par représentation string pour être déterministe
    pset = list(frame.get_power_set())
    # Note: On s'assure que l'ensemble vide n'est PAS inclus si get_power_set l'exclut
    # Jousselme travaille généralement sur 2^Theta \ {vide} pour la distance
    return sorted(pset, key=lambda x: (len(x), str(sorted(list(x)))))

def compute_jousselme_matrix(ordered_pset: List[FrozenSet]) -> np.ndarray:
    n = len(ordered_pset)
    D = np.zeros((n, n))
    for i, A in enumerate(ordered_pset):
        for j, B in enumerate(ordered_pset):
            if not A and not B: # Cas vide/vide (ne devrait pas arriver si exclu)
                D[i,j] = 1.0
                continue
            
            inter = len(A & B)
            union = len(A | B)
            D[i, j] = inter / union if union > 0 else 0.0
    return D

def mass_to_vector(mass: 'Mass', ordered_pset: List[FrozenSet]) -> np.ndarray:
    return np.array([mass.get_mass(s) for s in ordered_pset])

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
    
    def combine_two(self, m1: 'Mass', m2: 'Mass') -> 'Mass':
        result = Mass(m1.frame, f"{m1.name}⊕{m2.name}")
        assignments = defaultdict(float)
        K = 0.0
        
        # Récupération sécurisée des masses
        masses1 = m1.get_all_masses()
        masses2 = m2.get_all_masses()
        
        for B, m1_val in masses1.items():
            for C, m2_val in masses2.items():
                intersection = B & C
                product = m1_val * m2_val
                
                if len(intersection) == 0:
                    K += product
                else:
                    assignments[intersection] += product
        
        # Gestion de la tolérance numérique pour K
        if K > 1.0 - 1e-10:
             # Cas limite : conflit quasi total
             # Pour éviter le crash dans les itérations Murphy si le conflit est très élevé
             # on peut retourner une masse vacueuse ou lever une erreur
             raise ValueError(f"Conflit total détecté (K={K})")
             
        denom = 1.0 - K
        for subset, value in assignments.items():
            result.set_mass(subset, value / denom)
        
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

class DengCombination(CombinationRule):
    """
    Règle de Deng (2004) : Moyenne pondérée par distance Jousselme.
    Support = Somme des similarités (SANS diagonale, i.e., j != i).
    """
    def __init__(self):
        super().__init__("Deng")
        self.dempster = DempsterCombination()

    def combine(self, masses: List['Mass']) -> 'Mass':
        if not masses: raise ValueError("Aucune masse")
        if len(masses) == 1: return masses[0].copy()
        
        # 1. Préparation Jousselme
        pset = get_ordered_powerset(masses[0].frame)
        D = compute_jousselme_matrix(pset)
        vectors = np.array([mass_to_vector(m, pset) for m in masses])
        k = len(masses)
        
        # 2. Matrice de distance et similarité
        # Distance Jousselme : d_ij = sqrt(0.5 * (mi-mj)^T * D * (mi-mj))
        SIM = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    SIM[i, j] = 1.0
                    continue
                diff = vectors[i] - vectors[j]
                # Calcul matriciel optimisé : v @ D @ v.T
                dist_sq = 0.5 * (diff @ D @ diff.T)
                # Protection racine carrée négative due aux flottants
                dist = np.sqrt(max(0.0, dist_sq))
                SIM[i, j] = 1.0 - dist
        
        # 3. Poids (Deng: j != i)
        # On soustrait la diagonale (qui vaut 1) de la somme de la ligne
        Sup = np.sum(SIM, axis=1) - 1.0
        
        sum_sup = np.sum(Sup)
        if sum_sup == 0:
            Crd = np.ones(k) / k
        else:
            Crd = Sup / sum_sup
            
        # 4. Moyenne pondérée
        avg_mass = Mass(masses[0].frame, "Average_Deng")
        # Union de toutes les focales
        all_focals = set().union(*[m.get_all_masses().keys() for m in masses])
        
        for focal in all_focals:
            val = sum(Crd[i] * masses[i].get_mass(focal) for i in range(k))
            if val > 0:
                avg_mass.set_mass(focal, val)
                
        # 5. Fusion N-1 fois (Murphy)
        result = avg_mass.copy()
        for _ in range(k - 1):
            result = self.dempster.combine_two(result, avg_mass)
            
        self._preserve_explicit_zeros(result, masses)
        return result

class ZhangCombination(CombinationRule):
    """
    Règle de Zhang (2014) : Moyenne pondérée par Cosinus sur Pignistic.
    Support = Somme des similarités (AVEC diagonale, i.e., incluant i=i).
    """
    def __init__(self):
        super().__init__("Zhang")
        self.dempster = DempsterCombination()

    def get_pignistic_vector(self, mass: 'Mass') -> np.ndarray:
        # L'ordre doit être celui de frame.hypotheses
        # frame.hypotheses doit être une liste ordonnée des singletons (ex: ['A', 'B', 'C'])
        hyp_list = list(mass.frame.hypotheses) # Assurer liste stable
        vec = np.zeros(len(hyp_list))
        
        for subset, val in mass.get_all_masses().items():
            if val > 0 and len(subset) > 0:
                split_val = val / len(subset)
                for elem in subset:
                    try:
                        idx = hyp_list.index(elem)
                        vec[idx] += split_val
                    except ValueError:
                        pass # Élément inconnu du frame ?
        return vec

    def combine(self, masses: List['Mass']) -> 'Mass':
        if not masses: raise ValueError("Aucune masse")
        if len(masses) == 1: return masses[0].copy()
        
        k = len(masses)
        pig_vectors = [self.get_pignistic_vector(m) for m in masses]
        
        # 1. Matrice de similarité Cosinus
        SIM = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                # Dot product
                dot = np.dot(pig_vectors[i], pig_vectors[j])
                norm_i = np.linalg.norm(pig_vectors[i])
                norm_j = np.linalg.norm(pig_vectors[j])
                
                if norm_i > 0 and norm_j > 0:
                    SIM[i, j] = dot / (norm_i * norm_j)
                else:
                    SIM[i, j] = 0.0 # Cas pathologique masse nulle
        
        # 2. Support (Zhang: AVEC diagonale)
        # Le papier dit "correlation matrix S... elements on diagonal are 1"
        # "Sum of elements in row i is degree of support"
        Sup = np.sum(SIM, axis=1)
        
        sum_sup = np.sum(Sup)
        if sum_sup == 0:
            Crd = np.ones(k) / k
        else:
            Crd = Sup / sum_sup
            
        # 3. Moyenne pondérée
        avg_mass = Mass(masses[0].frame, "Average_Zhang")
        all_focals = set().union(*[m.get_all_masses().keys() for m in masses])
        
        for focal in all_focals:
            val = sum(Crd[i] * masses[i].get_mass(focal) for i in range(k))
            if val > 0:
                avg_mass.set_mass(focal, val)
                
        # 4. Fusion N-1 fois
        result = avg_mass.copy()
        for _ in range(k - 1):
            result = self.dempster.combine_two(result, avg_mass)
            
        self._preserve_explicit_zeros(result, masses)
        return result

class HanCombination(CombinationRule):
    """
    Règle de Han (2011) : Distance Jousselme + Mesure d'Incertitude (AM).
    Poids modifiés par une fonction puissance de l'entropie normalisée.
    """
    def __init__(self):
        super().__init__("Han")
        self.dempster = DempsterCombination()

    def get_ambiguity_measure(self, mass: 'Mass') -> float:
        # AM(m) = - sum( BetP(x) * log2(BetP(x)) )
        # On réutilise la logique pignistique (locale pour éviter dépendance)
        hyp_list = list(mass.frame.hypotheses)
        vec = np.zeros(len(hyp_list))
        
        for subset, val in mass.get_all_masses().items():
            if val > 0 and len(subset) > 0:
                split = val / len(subset)
                for elem in subset:
                    vec[hyp_list.index(elem)] += split
                    
        am = 0.0
        for p in vec:
            if p > 1e-10: # Éviter log(0)
                am -= p * np.log2(p)
        return am

    def combine(self, masses: List['Mass']) -> 'Mass':
        if not masses: raise ValueError("Aucune masse")
        k = len(masses)
        
        # 1. Poids de base (comme Deng)
        # Han utilise la distance pour le Cred initial
        pset = get_ordered_powerset(masses[0].frame)
        D = compute_jousselme_matrix(pset)
        vectors = np.array([mass_to_vector(m, pset) for m in masses])
        
        SIM = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j: 
                    SIM[i, j] = 1.0
                    continue
                diff = vectors[i] - vectors[j]
                dist = np.sqrt(max(0.0, 0.5 * (diff @ D @ diff.T)))
                SIM[i, j] = 1.0 - dist
                
        # Han utilise le support SANS diagonale pour le Cred initial (comme Deng)
        Sup = np.sum(SIM, axis=1) - 1.0
        sum_sup = np.sum(Sup)
        Cred = Sup / sum_sup if sum_sup > 0 else np.ones(k)/k
        
        # 2. Mesure d'Incertitude (AM)
        AM = np.array([self.get_ambiguity_measure(m) for m in masses])
        sum_am = np.sum(AM)
        # Normalisation Ent(mi)
        Ent = AM / sum_am if sum_am > 0 else np.ones(k)/k
        
        # 3. Modification des poids (Formule 12 corrigée)
        # DeltaCred = Cred(mi) - Average(Cred)
        # Credm = Cred * Ent^(-DeltaCred)
        avg_cred = 1.0 / k # Car somme Cred = 1
        Delta_Cred = Cred - avg_cred
        
        Cred_m = np.zeros(k)
        for i in range(k):
            # Attention : Ent[i] peut être 0 si masse certaine (ex: m({A})=1)
            # 0^(-x) peut poser problème.
            # Han ne spécifie pas le cas Ent=0, mais une certitude absolue est "bonne".
            # Si Ent ~ 0, on peut le clipper à epsilon ou traiter à part.
            base_ent = max(Ent[i], 1e-9) 
            exponent = -Delta_Cred[i]
            Cred_m[i] = Cred[i] * (base_ent ** exponent)
            
        # Normalisation finale des poids modifiés
        sum_cred_m = np.sum(Cred_m)
        Final_Weights = Cred_m / sum_cred_m if sum_cred_m > 0 else np.ones(k)/k
        
        # 4. Moyenne pondérée
        avg_mass = Mass(masses[0].frame, "Average_Han")
        all_focals = set().union(*[m.get_all_masses().keys() for m in masses])
        
        for focal in all_focals:
            val = sum(Final_Weights[i] * masses[i].get_mass(focal) for i in range(k))
            if val > 0:
                avg_mass.set_mass(focal, val)
                
        # 5. Fusion N-1 fois
        result = avg_mass.copy()
        for _ in range(k - 1):
            result = self.dempster.combine_two(result, avg_mass)
            
        self._preserve_explicit_zeros(result, masses)
        return result


# Registry des règles disponibles
COMBINATION_RULES = {
    "Dempster-Shafer": DempsterCombination(),
    "Smets (TBM)": SmetsCombination(),
    "Yager": YagerCombination(),
    "Murphy": MurphyCombination(),
    "PCR6": PCR6Combination(),
    "Deng": DengCombination(),
    "Zhang": ZhangCombination(),
    "Han": HanCombination()
}


def get_rule(name: str) -> CombinationRule:
    """Récupère une règle par son nom"""
    if name not in COMBINATION_RULES:
        raise ValueError(f"Règle inconnue: {name}")
    return COMBINATION_RULES[name]