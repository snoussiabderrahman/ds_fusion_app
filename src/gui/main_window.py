"""
Fenêtre principale remaniée — look professionnel, menu, toolbar, splitter, statusbar
"""
import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QFileDialog,
    QSplitter,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAction,
    QToolBar,
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon
from .styles import MAIN_STYLE
from .widgets import ResultDisplayWidget, MassCardWidget
from ..core.frame import FrameOfDiscernment
from ..core.mass import Mass
from ..core.combinations import COMBINATION_RULES, get_rule
from ..utils.parser import parse_mass_assignment, parse_fusion_expression
from ..utils.io_handler import (
    export_masses_to_csv,
    export_result_to_csv,
    import_masses_from_csv,
)
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.masses = {}
        self.current_result = None
        self.num_masses = 2

        self.setWindowTitle("Fusion Dempster-Shafer — DS Fusion")
        self.setGeometry(80, 80, 1400, 900)
        self.setStyleSheet(MAIN_STYLE)

        self.init_ui()

    def init_ui(self):
        # Menu bar and toolbar
        self.create_menu_and_toolbar()

        # Create main panels and use a resizable splitter
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()

        # Central widget with margins
        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(12, 12, 12, 12)
        central_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setHandleWidth(6)

        central_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

        # Status bar
        #self.statusBar().showMessage("Prêt")

        # Auto-fill initial fusion expression
        self.auto_fill_fusion()

    def create_menu_and_toolbar(self):
        # --- Menu ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Fichier")

        import_action = QAction("Importer...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_masses)
        file_menu.addAction(import_action)

        export_action = QAction("Exporter masses...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_masses)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        exit_action = QAction("Quitter", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        reset_menu = menubar.addMenu("&Réinitialiser")
        reset_action = QAction("Réinitialiser", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.clear_all_masses)
        reset_menu.addAction(reset_action)

        help_menu = menubar.addMenu("&Aide")
        about_action = QAction("À propos", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)


    def show_about(self):
        QMessageBox.information(
            self,
            "À propos",
            "DS Fusion — Outil de fusion Dempster-Shafer et des autres règles de combinaison\n"
            "Auteur : SNOUSSI ABDERRAHMEN",
        )

    # ---------------------------
    # Panels / UI components
    # ---------------------------
    def create_left_panel(self):
        """Panneau gauche : définition et affichage des masses"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Definition group
        mass_def_group = QGroupBox("Définition des masses")
        mass_def_layout = QVBoxLayout()

        instructions = QLabel(
            "Saisir l'assignation (ex: m1(A,B) = 0.5). Utilisez '.' comme séparateur décimal."
        )
        instructions.setStyleSheet("color: #556070;")
        mass_def_layout.addWidget(instructions)

        input_layout = QHBoxLayout()
        self.mass_input = QLineEdit()
        self.mass_input.setPlaceholderText("Ex : m1(A) = 0.6")
        self.mass_input.setMinimumHeight(34)
        self.mass_input.returnPressed.connect(self.add_mass)
        input_layout.addWidget(self.mass_input)

        self.add_mass_button = QPushButton("Insérer")
        self.add_mass_button.setObjectName("addButton")
        self.add_mass_button.setFixedWidth(110)
        self.add_mass_button.clicked.connect(self.add_mass)
        input_layout.addWidget(self.add_mass_button)

        mass_def_layout.addLayout(input_layout)

        self.clear_all_button = QPushButton("Effacer toutes les masses")
        self.clear_all_button.setObjectName("deleteButton")
        self.clear_all_button.clicked.connect(self.clear_all_masses)
        mass_def_layout.addWidget(self.clear_all_button)

        mass_def_group.setLayout(mass_def_layout)
        layout.addWidget(mass_def_group, 0)

        # Masses display group (scrollable)
        masses_display_group = QGroupBox("Masses définies")
        masses_display_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(420)

        self.mass_cards_widget = QWidget()
        self.mass_cards_layout = QVBoxLayout(self.mass_cards_widget)
        self.mass_cards_layout.setSpacing(10)
        self.mass_cards_layout.addStretch()

        scroll.setWidget(self.mass_cards_widget)
        masses_display_layout.addWidget(scroll)
        masses_display_group.setLayout(masses_display_layout)
        layout.addWidget(masses_display_group, 1)

        # IO buttons
        io_layout = QHBoxLayout()
        self.import_button = QPushButton("Importer (CSV)")
        self.import_button.clicked.connect(self.import_masses)
        self.export_button = QPushButton("Exporter (CSV)")
        self.export_button.clicked.connect(self.export_masses)
        io_layout.addWidget(self.import_button)
        io_layout.addWidget(self.export_button)
        layout.addLayout(io_layout)

        return widget

    def create_right_panel(self):
        """Panneau droit : configuration, fusion et résultats"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Frame / hypotheses
        frame_group = QGroupBox("Cadre de discernement (Θ)")
        frame_layout = QVBoxLayout()
        card_layout = QHBoxLayout()
        card_layout.addWidget(QLabel("Cardinalité (hypothèses):"))

        self.cardinality_input = QSpinBox()
        self.cardinality_input.setMinimum(1)
        self.cardinality_input.setMaximum(26)
        self.cardinality_input.setValue(3)
        self.cardinality_input.setFixedWidth(90)
        card_layout.addWidget(self.cardinality_input)

        self.set_frame_button = QPushButton("Définir")
        self.set_frame_button.clicked.connect(self.set_frame)
        card_layout.addWidget(self.set_frame_button)
        card_layout.addStretch()
        frame_layout.addLayout(card_layout)

        self.frame_label = QLabel("Non défini")
        self.frame_label.setStyleSheet("color: #c53030; font-style: italic;")
        frame_layout.addWidget(self.frame_label)
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)

        # Number of masses
        num_masses_group = QGroupBox("Nombre de masses")
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Masses à définir:"))

        self.num_masses_input = QSpinBox()
        self.num_masses_input.setMinimum(1)
        self.num_masses_input.setMaximum(10)
        self.num_masses_input.setValue(2)
        self.num_masses_input.setFixedWidth(90)
        self.num_masses_input.valueChanged.connect(self.on_num_masses_changed)
        num_layout.addWidget(self.num_masses_input)
        num_layout.addStretch()

        self.num_masses_label = QLabel("m1, m2")
        self.num_masses_label.setStyleSheet("color: #1f5fbf; font-weight: 600;")
        num_layout.addWidget(self.num_masses_label)

        num_masses_group.setLayout(num_layout)
        layout.addWidget(num_masses_group)

        # Fusion group
        fusion_group = QGroupBox("Fusion")
        fusion_layout = QVBoxLayout()

        rule_layout = QHBoxLayout()
        rule_layout.addWidget(QLabel("Règle:"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(COMBINATION_RULES.keys())
        rule_layout.addWidget(self.rule_combo)
        rule_layout.addStretch()
        fusion_layout.addLayout(rule_layout)

        expr_layout = QHBoxLayout()
        self.fusion_input = QLineEdit()
        self.fusion_input.setPlaceholderText("m1 + m2 + ...")
        expr_layout.addWidget(self.fusion_input)
        self.auto_fill_button = QPushButton("Auto")
        self.auto_fill_button.setFixedWidth(70)
        self.auto_fill_button.clicked.connect(self.auto_fill_fusion)
        expr_layout.addWidget(self.auto_fill_button)
        fusion_layout.addLayout(expr_layout)

        help_text = QLabel(
            "Séparez les masses par '+', ex: m1 + m2 + m3. Raccourci: Ctrl+F pour fusion."
        )
        help_text.setStyleSheet("color: #556070; font-size: 11px;")
        fusion_layout.addWidget(help_text)

        self.fusion_button = QPushButton("FUSIONNER")
        self.fusion_button.clicked.connect(self.perform_fusion)
        fusion_layout.addWidget(self.fusion_button)

        fusion_group.setLayout(fusion_layout)
        layout.addWidget(fusion_group)

        # Results
        result_group = QGroupBox("Résultat de fusion")
        result_layout = QVBoxLayout()
        self.result_display = ResultDisplayWidget()
        result_layout.addWidget(self.result_display)

        self.export_result_button = QPushButton("Exporter résultat (CSV)")
        self.export_result_button.clicked.connect(self.export_result)
        result_layout.addWidget(self.export_result_button)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group, 1)

        return widget

    # ---------------------------
    # Handlers (majority logic unchanged)
    # ---------------------------
    def on_num_masses_changed(self, value):
        self.num_masses = value
        mass_names = [f"m{i+1}" for i in range(value)]
        self.num_masses_label.setText(", ".join(mass_names))
        self.auto_fill_fusion()

    def auto_fill_fusion(self):
        mass_names = [f"m{i+1}" for i in range(self.num_masses)]
        self.fusion_input.setText(" + ".join(mass_names))

    def set_frame(self):
        try:
            cardinality = self.cardinality_input.value()
            if cardinality > 10:
                reply = QMessageBox.question(
                    self,
                    "Confirmation",
                    f"Une cardinalité de {cardinality} peut entraîner des lenteurs.\n"
                    "Voulez-vous continuer?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return

            self.frame = FrameOfDiscernment.from_cardinality(cardinality)
            hypotheses_str = "{" + ", ".join(self.frame.hypotheses) + "}"
            self.frame_label.setText(f"✓ Θ = {hypotheses_str}")
            self.frame_label.setStyleSheet("color: #27ae60; font-weight: bold;")

            # Reset
            self.masses = {}
            self.update_mass_cards()
            self.result_display.clear()
            self.current_result = None

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur: {e}")

    def add_mass(self):
        if not self.frame:
            QMessageBox.warning(self, "Attention", "Veuillez définir le cadre.")
            return

        try:
            text = self.mass_input.text().strip()
            if not text:
                return

            mass_name, subset_str, value = parse_mass_assignment(text)

            expected_masses = [f"m{i+1}" for i in range(self.num_masses)]
            if mass_name not in expected_masses:
                QMessageBox.critical(
                    self,
                    "Erreur",
                    f"La masse '{mass_name}' n'est pas valide.\nMasses attendues: "
                    f"{', '.join(expected_masses)}",
                )
                return

            subset = self.frame.parse_subset(subset_str)

            if mass_name not in self.masses:
                self.masses[mass_name] = Mass(self.frame, mass_name)

            self.masses[mass_name].set_mass(subset, value)

            self.update_mass_cards()
            self.mass_input.clear()
            self.mass_input.setFocus()

        except Exception as e:
            QMessageBox.critical(self, "Format invalide", str(e))

    def clear_all_masses(self):
        if not self.masses:
            return
        reply = QMessageBox.question(
            self,
            "Confirmation",
            "Voulez-vous effacer toutes les masses définies?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.masses = {}
            self.update_mass_cards()
            self.result_display.clear()
            self.current_result = None

    def delete_mass(self, mass_name):
        if mass_name in self.masses:
            del self.masses[mass_name]
            self.update_mass_cards()

    def update_mass_cards(self):
        # clear existing
        for i in reversed(range(self.mass_cards_layout.count())):
            widget = self.mass_cards_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for mass_name in sorted(self.masses.keys()):
            mass = self.masses[mass_name]
            card = MassCardWidget(mass_name, mass, self.delete_mass)
            self.mass_cards_layout.insertWidget(
                self.mass_cards_layout.count() - 1, card
            )

    def perform_fusion(self):
        if not self.frame or not self.masses:
            QMessageBox.warning(self, "Attention", "Définir le cadre et au moins une masse.")
            return

        try:
            expression = self.fusion_input.text()
            if not expression.strip():
                QMessageBox.warning(self, "Attention", "Saisir une expression de fusion.")
                return

            mass_names = parse_fusion_expression(expression)
            masses_to_fuse = []
            for name in mass_names:
                if name not in self.masses:
                    raise ValueError(f"La masse '{name}' n'est pas définie.")
                masses_to_fuse.append(self.masses[name])

            invalid_masses = []
            for mass in masses_to_fuse:
                if not mass.is_valid():
                    invalid_masses.append(f"{mass.name}: somme={mass.get_total():.4f}")

            if invalid_masses:
                msg = "Les masses suivantes ont une somme ≠ 1.0:\n\n"
                msg += "\n".join(invalid_masses)
                msg += "\n\nVoulez-vous les normaliser automatiquement?"
                reply = QMessageBox.question(self, "Masses non valides", msg, QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    for mass in masses_to_fuse:
                        mass.normalize()
                    self.update_mass_cards()
                else:
                    return

            rule_name = self.rule_combo.currentText()
            rule = get_rule(rule_name)

            self.current_result = rule.combine(masses_to_fuse)
            self.result_display.display_result(self.current_result, rule_name, expression)
            self.statusBar().showMessage(f"Fusion effectuée ({rule_name})", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Erreur de fusion", str(e))

    def export_masses(self):
        if not self.masses:
            QMessageBox.warning(self, "Attention", "Aucune masse à exporter.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Exporter les masses", "", "CSV (*.csv)")
        if filename:
            try:
                export_masses_to_csv(self.masses, filename)
                QMessageBox.information(self, "Succès", f"Masses exportées: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur export", str(e))

    def export_result(self):
        if not self.current_result:
            QMessageBox.warning(self, "Attention", "Aucun résultat à exporter.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Exporter le résultat", "", "CSV (*.csv)")
        if filename:
            try:
                export_result_to_csv(self.current_result, filename)
                QMessageBox.information(self, "Succès", f"Résultat exporté: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur export", str(e))

    def import_masses(self):
        if not self.frame:
            QMessageBox.warning(self, "Attention", "Veuillez définir le cadre avant d'importer.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Importer des masses", "", "CSV (*.csv)")
        if filename:
            try:
                imported_masses = import_masses_from_csv(filename, self.frame)
                self.masses.update(imported_masses)
                self.update_mass_cards()
                QMessageBox.information(self, "Succès", f"Masses importées depuis : {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur import", str(e))