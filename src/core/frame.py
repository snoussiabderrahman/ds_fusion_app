from itertools import chain, combinations
from typing import Set, FrozenSet, List


class FrameOfDiscernment:
    """Cadre de discernement Θ"""
    
    def __init__(self, hypotheses: List[str]):
        """
        Args:
            hypotheses: Liste des hypothèses (ex: ['A', 'B', 'C'])
        """
        self.hypotheses = sorted(hypotheses)
        self.n = len(hypotheses)
        self._power_set = None
    
    @classmethod
    def from_cardinality(cls, cardinality: int):
        """Crée un frame avec cardinalité donnée"""
        hypotheses = [chr(65 + i) for i in range(cardinality)]  # A, B, C, ...
        return cls(hypotheses)
    
    def get_power_set(self) -> List[FrozenSet[str]]:
        """Retourne l'ensemble des parties (power set) sans l'ensemble vide"""
        if self._power_set is None:
            self._power_set = []
            for r in range(1, self.n + 1):
                for combo in combinations(self.hypotheses, r):
                    self._power_set.append(frozenset(combo))
        return self._power_set
    
    def get_theta(self) -> FrozenSet[str]:
        """Retourne Θ (cadre complet)"""
        return frozenset(self.hypotheses)
    
    def format_subset(self, subset: FrozenSet[str]) -> str:
        """Formate un sous-ensemble pour l'affichage"""
        if not subset:
            return "∅"
        if subset == self.get_theta():
            return "Θ"
        return "{" + ",".join(sorted(subset)) + "}"
    
    def parse_subset(self, text: str) -> FrozenSet[str]:
        """Parse une chaîne en sous-ensemble"""
        text = text.strip()
        
        if text in ['∅', 'Ø', 'empty', '{}']:
            return frozenset()
        
        if text in ['Θ', 'Omega', 'theta', 'THETA']:
            return self.get_theta()
        
        # Enlever les accolades si présentes
        text = text.strip('{}')
        
        # Séparer par virgules
        elements = [e.strip() for e in text.split(',')]
        
        # Valider que tous les éléments sont dans le frame
        for elem in elements:
            if elem not in self.hypotheses:
                raise ValueError(f"'{elem}' n'est pas dans le cadre de discernement")
        
        return frozenset(elements)
    
    def __repr__(self):
        return f"Frame({self.hypotheses})"