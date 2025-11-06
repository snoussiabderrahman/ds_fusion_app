"""
Fenêtre principale de l'application
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QSpinBox,
                             QLineEdit, QPushButton, QComboBox, QMessageBox,
                             QFileDialog, QSplitter, QScrollArea, QGridLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from .styles import MAIN_STYLE
from .widgets import  ResultDisplayWidget, MassCardWidget
from ..core.frame import FrameOfDiscernment
from ..core.mass import Mass
from ..core.combinations import COMBINATION_RULES, get_rule
from ..utils.parser import parse_mass_assignment, parse_fusion_expression
from ..utils.io_handler import export_masses_to_csv, export_result_to_csv, import_masses_from_csv
import numpy as np


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.frame = None
        self.masses = {}
        self.current_result = None
        self.num_masses = 2

        
        self.setWindowTitle("Application de Fusion Dempster-Shafer")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(MAIN_STYLE)
        
        self.init_ui()
    
    def init_ui(self):
        # Widget central et layout principal
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- PARTIE GAUCHE : Définition et affichage des masses ---
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 45)  # 45% de l'espace
        
        # --- PARTIE DROITE : Configuration, fusion et résultats ---
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 55)  # 55% de l'espace
        
        self.setCentralWidget(central_widget)
    
    def create_left_panel(self):
        """Crée le panneau de gauche pour gérer les masses"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # --- Groupe Définition des masses (20%) ---
        mass_def_group = QGroupBox("3. Définir les Masses")
        mass_def_layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("💡 Saisir l'assignation (ex: m1(A,B) = 0.5 ou m1(A) = 0)")
        instructions.setStyleSheet("color: #7f8c8d; font-size: 13px;")
        mass_def_layout.addWidget(instructions)
        
        # Input et bouton
        input_layout = QHBoxLayout()
        self.mass_input = QLineEdit()
        self.mass_input.setPlaceholderText("m1(A) = 0.6")
        self.mass_input.setMinimumHeight(35)
        self.mass_input.returnPressed.connect(self.add_mass)
        input_layout.addWidget(self.mass_input)
        
        self.add_mass_button = QPushButton("✓ Insérer")
        self.add_mass_button.setObjectName("addButton")
        self.add_mass_button.setFixedWidth(120)
        self.add_mass_button.setMinimumHeight(35)
        self.add_mass_button.clicked.connect(self.add_mass)
        input_layout.addWidget(self.add_mass_button)
        
        mass_def_layout.addLayout(input_layout)
        
        # Bouton effacer
        self.clear_all_button = QPushButton("🗑 Effacer Toutes les Masses")
        self.clear_all_button.setObjectName("deleteButton")
        self.clear_all_button.setMinimumHeight(32)
        self.clear_all_button.clicked.connect(self.clear_all_masses)
        mass_def_layout.addWidget(self.clear_all_button)
        
        mass_def_group.setLayout(mass_def_layout)
        layout.addWidget(mass_def_group, 10)  # 10% de l'espace
        
        # --- Affichage des masses par cartes (80%) ---
        masses_display_group = QGroupBox("📋 Masses Définies")
        masses_display_layout = QVBoxLayout()
        
        # Zone scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)
        
        self.mass_cards_widget = QWidget()
        self.mass_cards_layout = QVBoxLayout(self.mass_cards_widget)
        self.mass_cards_layout.setSpacing(12)
        self.mass_cards_layout.addStretch()
        
        scroll.setWidget(self.mass_cards_widget)
        masses_display_layout.addWidget(scroll)
        
        masses_display_group.setLayout(masses_display_layout)
        layout.addWidget(masses_display_group, 90)  # 90% de l'espace
        
        # Boutons Import/Export
        io_layout = QHBoxLayout()
        self.import_button = QPushButton("📥 Importer (CSV)")
        self.import_button.setMinimumHeight(35)
        self.import_button.clicked.connect(self.import_masses)
        self.export_button = QPushButton("📤 Exporter (CSV)")
        self.export_button.setObjectName("exportButton")
        self.export_button.setMinimumHeight(35)
        self.export_button.clicked.connect(self.export_masses)
        io_layout.addWidget(self.import_button)
        io_layout.addWidget(self.export_button)
        layout.addLayout(io_layout)
        
        return widget
    
    def create_right_panel(self):
        """Crée le panneau de droite"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # --- 1. Cadre de Discernement (10%) ---
        frame_group = QGroupBox("1. Cadre de Discernement (Θ)")
        frame_layout = QVBoxLayout()
        
        card_layout = QHBoxLayout()
        card_layout.addWidget(QLabel("Cardinalité (hypothèses):"))
        
        self.cardinality_input = QSpinBox()
        self.cardinality_input.setMinimum(1)
        self.cardinality_input.setMaximum(26)
        self.cardinality_input.setValue(3)
        self.cardinality_input.setFixedWidth(80)
        self.cardinality_input.setMinimumHeight(35)
        self.cardinality_input.setButtonSymbols(QSpinBox.UpDownArrows)
        card_layout.addWidget(self.cardinality_input)
        
        self.set_frame_button = QPushButton("Définir")
        self.set_frame_button.setFixedWidth(120)
        self.set_frame_button.setMinimumHeight(35)
        self.set_frame_button.clicked.connect(self.set_frame)
        card_layout.addWidget(self.set_frame_button)
        card_layout.addStretch()
        
        frame_layout.addLayout(card_layout)
        
        self.frame_label = QLabel("⚠ Non défini")
        self.frame_label.setStyleSheet("color: #e74c3c; font-style: italic; padding: 5px;")
        frame_layout.addWidget(self.frame_label)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group, 10)  # 10%
        
        # --- 2. Nombre de Masses (10%) ---
        num_masses_group = QGroupBox("2. Nombre de Masses")
        num_layout = QVBoxLayout()
        
        num_input_layout = QHBoxLayout()
        num_input_layout.addWidget(QLabel("Masses à définir:"))
        
        self.num_masses_input = QSpinBox()
        self.num_masses_input.setMinimum(1)
        self.num_masses_input.setMaximum(10)
        self.num_masses_input.setValue(2)
        self.num_masses_input.setFixedWidth(80)
        self.num_masses_input.setMinimumHeight(35)
        self.num_masses_input.setButtonSymbols(QSpinBox.UpDownArrows)
        self.num_masses_input.valueChanged.connect(self.on_num_masses_changed)
        num_input_layout.addWidget(self.num_masses_input)
        
        num_input_layout.addStretch()
        num_layout.addLayout(num_input_layout)
        
        self.num_masses_label = QLabel("📊 m1, m2")
        self.num_masses_label.setStyleSheet(
            "color: #27ae60; font-weight: bold; padding: 5px;"
        )
        num_layout.addWidget(self.num_masses_label)
        
        num_masses_group.setLayout(num_layout)
        layout.addWidget(num_masses_group, 10)  # 10%
        
        # --- 3. Fusion (20%) ---
        fusion_group = QGroupBox("4. Fusionner les Masses")
        fusion_layout = QVBoxLayout()
        
        # Règle de combinaison
        rule_layout = QHBoxLayout()
        rule_layout.addWidget(QLabel("Règle de combinaison:"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(COMBINATION_RULES.keys())
        self.rule_combo.setMinimumHeight(32)
        rule_layout.addWidget(self.rule_combo)
        fusion_layout.addLayout(rule_layout)
        
        # Expression de fusion
        expr_label = QLabel("Expression de fusion:")
        expr_label.setStyleSheet("margin-top: 8px;")
        fusion_layout.addWidget(expr_label)
        
        expr_layout = QHBoxLayout()
        self.fusion_input = QLineEdit()
        self.fusion_input.setPlaceholderText("m1 + m2")
        self.fusion_input.setMinimumHeight(35)
        self.fusion_input.returnPressed.connect(self.perform_fusion)
        expr_layout.addWidget(self.fusion_input)
        
        self.auto_fill_button = QPushButton("↻ Auto")
        self.auto_fill_button.setFixedWidth(80)
        self.auto_fill_button.setMinimumHeight(35)
        self.auto_fill_button.clicked.connect(self.auto_fill_fusion)
        expr_layout.addWidget(self.auto_fill_button)
        
        fusion_layout.addLayout(expr_layout)
        
        # Aide
        help_text = QLabel("💡 Séparez les masses par '+' (ex: m1 + m2 + m3)")
        help_text.setStyleSheet("color: #7f8c8d; font-size: 10px; margin-top: 3px;")
        fusion_layout.addWidget(help_text)
        
        self.fusion_button = QPushButton("FUSIONNER")
        self.fusion_button.setObjectName("fusionButton")
        self.fusion_button.clicked.connect(self.perform_fusion)
        self.fusion_button.setMinimumHeight(45)
        fusion_layout.addWidget(self.fusion_button)
        
        fusion_group.setLayout(fusion_layout)
        layout.addWidget(fusion_group, 20)  # 20%
        
        # --- 4. Résultats (60%) ---
        result_group = QGroupBox("📊 Résultats de Fusion")
        result_layout = QVBoxLayout()
        
        self.result_display = ResultDisplayWidget()
        result_layout.addWidget(self.result_display)
        
        self.export_result_button = QPushButton("📤 Exporter Résultat (CSV)")
        self.export_result_button.setObjectName("exportButton")
        self.export_result_button.setMinimumHeight(35)
        self.export_result_button.clicked.connect(self.export_result)
        result_layout.addWidget(self.export_result_button)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group, 60)  # 60%
        
        return widget
    
    def on_num_masses_changed(self, value):
        """Appelé quand le nombre de masses change"""
        self.num_masses = value
        mass_names = [f"m{i+1}" for i in range(value)]
        self.num_masses_label.setText("📊 " + ", ".join(mass_names))
        self.auto_fill_fusion()
    
    def auto_fill_fusion(self):
        """Remplit automatiquement l'expression de fusion"""
        mass_names = [f"m{i+1}" for i in range(self.num_masses)]
        self.fusion_input.setText(" + ".join(mass_names))
    
    def set_frame(self):
        """Définit le cadre de discernement"""
        try:
            cardinality = self.cardinality_input.value()
            if cardinality > 10:
                reply = QMessageBox.question(self, "Confirmation", 
                    f"Une cardinalité de {cardinality} peut entraîner des lenteurs.\n"
                    "Voulez-vous continuer?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    return
            
            self.frame = FrameOfDiscernment.from_cardinality(cardinality)
            hypotheses_str = "{" + ", ".join(self.frame.hypotheses) + "}"
            self.frame_label.setText(f"✓ Θ = {hypotheses_str}")
            self.frame_label.setStyleSheet(
                "color: #27ae60; font-weight: bold; padding: 5px;"
            )
            
            # Réinitialiser
            self.masses = {}
            self.update_mass_cards()
            self.result_display.clear()
            self.current_result = None
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur: {e}")
    
    def add_mass(self):
        """Ajoute ou modifie une assignation de masse"""
        if not self.frame:
            QMessageBox.warning(self, "Attention", 
                "Veuillez d'abord définir le cadre de discernement.")
            return
        
        try:
            text = self.mass_input.text().strip()
            if not text:
                return
            
            mass_name, subset_str, value = parse_mass_assignment(text)
            
            # VALIDATION : Vérifier que le nom de masse est valide
            expected_masses = [f"m{i+1}" for i in range(self.num_masses)]
            if mass_name not in expected_masses:
                QMessageBox.critical(self, "Erreur", 
                    f"La masse '{mass_name}' n'est pas valide.\n\n"
                    f"Masses attendues : {', '.join(expected_masses)}\n\n"
                    f"Vous avez défini {self.num_masses} masse(s). "
                    f"Utilisez uniquement ces noms.")
                return
            
            subset = self.frame.parse_subset(subset_str)
            
            # Créer la masse si elle n'existe pas
            if mass_name not in self.masses:
                self.masses[mass_name] = Mass(self.frame, mass_name)
            
            # Assigner la valeur (la classe Mass gère les zéros)
            self.masses[mass_name].set_mass(subset, value)
            
            self.update_mass_cards()
            self.mass_input.clear()
            self.mass_input.setFocus()
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Format invalide:\n{e}\n\n"
                "Utilisez le format: m1(A,B) = 0.5")
    
    def clear_all_masses(self):
        """Efface toutes les masses"""
        if not self.masses:
            return
        
        reply = QMessageBox.question(self, "Confirmation",
            "Voulez-vous vraiment effacer toutes les masses définies?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.masses = {}
            self.update_mass_cards()
            self.result_display.clear()
            self.current_result = None
    
    def delete_mass(self, mass_name):
        """Supprime une masse complète"""
        if mass_name in self.masses:
            del self.masses[mass_name]
            self.update_mass_cards()
    
    def update_mass_cards(self):
        """Met à jour l'affichage des cartes de masses"""
        # Supprimer toutes les cartes existantes
        for i in reversed(range(self.mass_cards_layout.count())): 
            widget = self.mass_cards_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Ajouter les cartes pour chaque masse
        for mass_name in sorted(self.masses.keys()):
            mass = self.masses[mass_name]
            card = MassCardWidget(mass_name, mass, self.delete_mass)
            self.mass_cards_layout.insertWidget(
                self.mass_cards_layout.count() - 1, card
            )
    
    def perform_fusion(self):
        """Effectue la fusion selon l'expression et la règle choisies"""
        if not self.frame or not self.masses:
            QMessageBox.warning(self, "Attention", 
                "Veuillez définir le frame et au moins une masse.")
            return
        
        try:
            expression = self.fusion_input.text()
            if not expression.strip():
                QMessageBox.warning(self, "Attention", 
                    "Veuillez saisir une expression de fusion.")
                return
            
            mass_names = parse_fusion_expression(expression)
            
            # Vérifier que les masses existent
            masses_to_fuse = []
            for name in mass_names:
                if name not in self.masses:
                    raise ValueError(f"La masse '{name}' n'est pas définie.")
                masses_to_fuse.append(self.masses[name])
            
            # Vérifier la validité des masses
            invalid_masses = []
            for mass in masses_to_fuse:
                if not mass.is_valid():
                    invalid_masses.append(
                        f"{mass.name}: somme = {mass.get_total():.4f}"
                    )
            
            if invalid_masses:
                msg = "Les masses suivantes ont une somme ≠ 1.0:\n\n"
                msg += "\n".join(invalid_masses)
                msg += "\n\nVoulez-vous les normaliser automatiquement?"
                
                reply = QMessageBox.question(self, "Masses non valides", msg,
                    QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    for mass in masses_to_fuse:
                        mass.normalize()
                    self.update_mass_cards()
                else:
                    return
            
            # Obtenir la règle
            rule_name = self.rule_combo.currentText()
            rule = get_rule(rule_name)
            
            # Effectuer la fusion
            self.current_result = rule.combine(masses_to_fuse)
            
            # Afficher le résultat
            self.result_display.display_result(
                self.current_result, rule_name, expression
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Fusion", str(e))
    
    def export_masses(self):
        """Exporte les masses définies vers un CSV"""
        if not self.masses:
            QMessageBox.warning(self, "Attention", "Aucune masse à exporter.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exporter les Masses", "", "Fichiers CSV (*.csv)"
        )
        if filename:
            try:
                export_masses_to_csv(self.masses, filename)
                QMessageBox.information(self, "Succès", 
                    f"Masses exportées vers:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur d'exportation:\n{e}")
    
    def export_result(self):
        """Exporte le résultat de la fusion vers un CSV"""
        if not self.current_result:
            QMessageBox.warning(self, "Attention", 
                "Aucun résultat de fusion à exporter.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exporter le Résultat", "", "Fichiers CSV (*.csv)"
        )
        if filename:
            try:
                export_result_to_csv(self.current_result, filename)
                QMessageBox.information(self, "Succès", 
                    f"Résultat exporté vers:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur d'exportation:\n{e}")
    
    def import_masses(self):
        """Importe des masses depuis un CSV"""
        if not self.frame:
            QMessageBox.warning(self, "Attention", 
                "Veuillez d'abord définir le cadre de discernement.")
            return
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Importer des Masses", "", "Fichiers CSV (*.csv)"
        )
        if filename:
            try:
                imported_masses = import_masses_from_csv(filename, self.frame)
                self.masses.update(imported_masses)
                self.update_mass_cards()
                QMessageBox.information(self, "Succès", 
                    f"Masses importées depuis:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur d'importation:\n{e}")