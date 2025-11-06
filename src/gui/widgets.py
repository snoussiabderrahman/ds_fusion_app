"""
Widgets personnalisés pour l'interface
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QTextEdit, QHeaderView, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor


class MassCardWidget(QFrame):
    """Widget en forme de carte pour afficher une masse"""
    
    def __init__(self, mass_name, mass, delete_callback, explicit_zeros=None):
        super().__init__()
        self.mass_name = mass_name
        self.mass = mass
        self.delete_callback = delete_callback
        # Utiliser les zéros explicites de la masse elle-même
        self.explicit_zeros = mass.get_explicit_zeros()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            MassCardWidget {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # En-tête
        header_layout = QHBoxLayout()
        
        title = QLabel(f"📊 {self.mass_name}")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title)
        
        # Indicateur de validité
        total = self.mass.get_total()
        is_valid = abs(total - 1.0) < 1e-6
        
        if is_valid:
            status = QLabel("✓ Valide")
            status.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            status = QLabel(f"⚠ Somme: {total:.3f}")
            status.setStyleSheet("color: #e67e22; font-weight: bold;")
        header_layout.addWidget(status)
        
        header_layout.addStretch()
        
        delete_btn = QPushButton("✕")
        delete_btn.setObjectName("deleteButton")
        delete_btn.setFixedSize(30, 30)
        delete_btn.setToolTip("Supprimer cette masse")
        delete_btn.clicked.connect(lambda: self.delete_callback(self.mass_name))
        header_layout.addWidget(delete_btn)
        
        layout.addLayout(header_layout)
        
        # Tableau des assignations
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Sous-ensemble", "Valeur"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        table.setColumnWidth(1, 100)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        
        # Obtenir tous les éléments définis
        all_elements = self.mass.get_all_defined_elements()
        
        table.setRowCount(len(all_elements))
        
        # Calculer la hauteur nécessaire
        row_height = 30
        header_height = 35
        total_height = min(header_height + (len(all_elements) * row_height) + 10, 300)
        table.setMinimumHeight(total_height)
        table.setMaximumHeight(total_height)
        
        for row, subset in enumerate(all_elements):
            # Sous-ensemble
            subset_str = self.mass.frame.format_subset(subset)
            subset_item = QTableWidgetItem(subset_str)
            subset_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 0, subset_item)
            
            # Valeur
            value = self.mass.get_mass(subset)
            value_item = QTableWidgetItem(f"{value:.4f}")
            value_item.setTextAlignment(Qt.AlignCenter)
            
            # Style différent pour les zéros explicites
            if subset in self.explicit_zeros and value == 0:
                value_item.setForeground(QColor("#95a5a6"))
                value_font = QFont()
                value_font.setItalic(True)
                value_item.setFont(value_font)
            else:
                value_font = QFont()
                value_font.setBold(True)
                value_item.setFont(value_font)
            
            table.setItem(row, 1, value_item)
        
        layout.addWidget(table)
        
        self.setLayout(layout)


class ResultDisplayWidget(QWidget):
    """Widget pour afficher les résultats de fusion avec tableau"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # En-tête avec informations
        self.info_frame = QFrame()
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #eaf2f8;
                border: 2px solid #3498db;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        info_layout = QVBoxLayout(self.info_frame)
        
        # Titre
        self.title_label = QLabel("📊 Résultat de Fusion")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #2c3e50; background: transparent; border: none;")
        info_layout.addWidget(self.title_label)
        
        # Informations détaillées
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #34495e; background: transparent; border: none; font-size: 11px;")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(self.info_frame)
        
        # Tableau des résultats
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Sous-ensemble", "Valeur", "Pourcentage"])
        
        # Configuration des colonnes
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.result_table.setColumnWidth(1, 120)
        self.result_table.setColumnWidth(2, 120)
        
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        
        layout.addWidget(self.result_table)
        
        # Message si pas de résultat
        self.empty_label = QLabel("🔍 Aucun résultat de fusion.\nEffectuez une fusion pour voir les résultats ici.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            color: #95a5a6;
            font-size: 13px;
            font-style: italic;
            padding: 40px;
        """)
        layout.addWidget(self.empty_label)
        
        self.setLayout(layout)
        self.clear()
    
    def display_result(self, result, rule_name, expression=""):
        """Affiche un résultat de fusion dans le tableau"""
        # Afficher les composants
        self.info_frame.setVisible(True)
        self.result_table.setVisible(True)
        self.empty_label.setVisible(False)
        
        # Mettre à jour le titre
        self.title_label.setText(f"📊 Résultat de Fusion - {rule_name}")
        
        # Mettre à jour les informations
        total = result.get_total()
        is_valid = result.is_valid()
        status_text = "✓ Valide" if is_valid else "✗ Non valide"
        status_color = "#27ae60" if is_valid else "#e74c3c"
        
        info_text = f"<b>Expression:</b> {expression}<br>"
        info_text += f"<b>Résultat:</b> {result.name}<br>"
        info_text += f"<b>Total:</b> {total:.6f} | "
        info_text += f"<b>Statut:</b> <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>"
        self.info_label.setText(info_text)
        
        # Remplir le tableau
        all_elements = result.get_all_defined_elements()
        self.result_table.setRowCount(len(all_elements))
        
        for row, subset in enumerate(all_elements):
            # Sous-ensemble
            subset_str = result.frame.format_subset(subset)
            subset_item = QTableWidgetItem(subset_str)
            subset_item.setTextAlignment(Qt.AlignCenter)
            subset_font = QFont()
            subset_font.setBold(True)
            subset_item.setFont(subset_font)
            self.result_table.setItem(row, 0, subset_item)
            
            # Valeur
            value = result.get_mass(subset)
            value_item = QTableWidgetItem(f"{value:.6f}")
            value_item.setTextAlignment(Qt.AlignCenter)
            value_font = QFont()
            value_font.setBold(True)
            value_item.setFont(value_font)
            
            # Colorer selon la valeur
            if value == 0:
                value_item.setForeground(QColor("#95a5a6"))
            elif value < 0.1:
                value_item.setForeground(QColor("#e67e22"))
            else:
                value_item.setForeground(QColor("#27ae60"))
            
            self.result_table.setItem(row, 1, value_item)
            
            # Pourcentage
            percentage = value * 100
            percent_item = QTableWidgetItem(f"{percentage:.2f} %")
            percent_item.setTextAlignment(Qt.AlignCenter)
            percent_font = QFont()
            percent_font.setItalic(True)
            percent_item.setFont(percent_font)
            
            if value == 0:
                percent_item.setForeground(QColor("#95a5a6"))
            elif value < 0.1:
                percent_item.setForeground(QColor("#e67e22"))
            else:
                percent_item.setForeground(QColor("#27ae60"))
            
            self.result_table.setItem(row, 2, percent_item)
    
    def clear(self):
        """Efface les résultats"""
        self.info_frame.setVisible(False)
        self.result_table.setVisible(False)
        self.empty_label.setVisible(True)
        self.result_table.setRowCount(0)