from typing import Dict, FrozenSet, Set
from .frame import FrameOfDiscernment


class Mass:
    """Fonction de masse (Basic Belief Assignment)"""
    
    def __init__(self, frame: FrameOfDiscernment, name: str = "m"):
        """
        Args:
            frame: Cadre de discernement
            name: Nom de la masse (ex: "m1", "m2")
        """
        self.frame = frame
        self.name = name
        self._assignments: Dict[FrozenSet[str], float] = {}
        # Pour garder trace des masses explicitement définies à 0
        self._explicit_zeros: Set[FrozenSet[str]] = set()
    
    def set_mass(self, subset: FrozenSet[str], value: float):
        """Assigne une masse à un sous-ensemble (tolérance pour erreurs numériques)."""
        eps = 1e-12  # tolérance pour arrondis flottants

        # Rejeter uniquement si on est nettement en dehors de [0,1]
        if value < -eps or value > 1.0 + eps:
            raise ValueError("La masse doit être entre 0 et 1")

        # Clamp / arrondir les petites imprécisions
        if abs(value) <= eps:
            value = 0.0
        elif abs(value - 1.0) <= eps:
            value = 1.0
        else:
            # Optionnel : réduire la précision pour éviter accumulation d'erreurs
            value = float(value)

        if value > 0:
            self._assignments[subset] = value
            # Retirer de explicit_zeros si c'était là
            self._explicit_zeros.discard(subset)
        elif value == 0:
            # Si on définit explicitement à 0, on l'enregistre
            self._explicit_zeros.add(subset)
            # Retirer de assignments si c'était là
            if subset in self._assignments:
                del self._assignments[subset]
        else:
            # Valeur négative (improbable ici), on retire tout
            if subset in self._assignments:
                del self._assignments[subset]
            self._explicit_zeros.discard(subset)

    
    def get_mass(self, subset: FrozenSet[str]) -> float:
        """Récupère la masse d'un sous-ensemble"""
        return self._assignments.get(subset, 0.0)
    
    def get_all_masses(self) -> Dict[FrozenSet[str], float]:
        """Retourne toutes les assignations de masse"""
        return dict(self._assignments)
    
    def get_all_masses_with_zeros(self) -> Dict[FrozenSet[str], float]:
        """Retourne toutes les assignations incluant les zéros explicites"""
        result = dict(self._assignments)
        for subset in self._explicit_zeros:
            if subset not in result:
                result[subset] = 0.0
        return result
    
    def has_explicit_zero(self, subset: FrozenSet[str]) -> bool:
        """Vérifie si un subset a été explicitement défini à 0"""
        return subset in self._explicit_zeros
    
    def get_explicit_zeros(self) -> Set[FrozenSet[str]]:
        """Retourne l'ensemble des masses explicitement définies à 0"""
        return set(self._explicit_zeros)
    
    def normalize(self):
        """Normalise la masse pour que la somme soit 1"""
        total = sum(self._assignments.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            for key in self._assignments:
                self._assignments[key] /= total
    
    def get_total(self) -> float:
        """Retourne la somme des masses"""
        return sum(self._assignments.values())
    
    def is_valid(self) -> bool:
        """Vérifie si la masse est valide (somme = 1)"""
        return abs(self.get_total() - 1.0) < 1e-6
    
    def get_focal_elements(self) -> list:
        """Retourne les éléments focaux (avec masse > 0)"""
        return sorted(self._assignments.keys(), 
                     key=lambda x: (len(x), sorted(x)))
    
    def get_all_defined_elements(self) -> list:
        """Retourne tous les éléments définis (focaux + zéros explicites)"""
        all_elements = set(self._assignments.keys()) | self._explicit_zeros
        return sorted(all_elements, key=lambda x: (len(x), sorted(x)))
    
    def copy(self):
        """Crée une copie de la masse"""
        new_mass = Mass(self.frame, self.name)
        new_mass._assignments = dict(self._assignments)
        new_mass._explicit_zeros = set(self._explicit_zeros)
        return new_mass
    
    def __repr__(self):
        items = []
        for subset in self.get_focal_elements():
            subset_str = self.frame.format_subset(subset)
            value = self._assignments[subset]
            items.append(f"{self.name}({subset_str})={value:.4f}")
        return ", ".join(items) if items else f"{self.name}(empty)"