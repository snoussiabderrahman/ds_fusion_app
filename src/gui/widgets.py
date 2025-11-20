from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QTextEdit,
    QHeaderView,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor


class MassCardWidget(QFrame):
    """Carte compacte et professionnelle pour une masse"""
    def __init__(self, mass_name, mass, delete_callback):
        super().__init__()
        self.setObjectName("massCard")
        self.mass_name = mass_name
        self.mass = mass
        self.delete_callback = delete_callback
        self.explicit_zeros = mass.get_explicit_zeros()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        title = QLabel(self.mass_name)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #1f2937;")
        header.addWidget(title)

        # validity badge
        total = self.mass.get_total()
        is_valid = abs(total - 1.0) < 1e-6
        status = QLabel()
        status_font = QFont()
        status_font.setPointSize(9)
        status_font.setBold(True)
        status.setFont(status_font)
        if is_valid:
            status.setText("Valide")
            status.setStyleSheet("color: #27ae60;")
        else:
            status.setText(f"Somme: {total:.3f}")
            status.setStyleSheet("color: #d97706;")
        header.addWidget(status)
        header.addStretch()

        # delete button
        delete_btn = QPushButton("Supprimer")
        delete_btn.setObjectName("deleteButton")
        delete_btn.setFixedHeight(26)
        delete_btn.setProperty("flat", True)
        delete_btn.clicked.connect(lambda: self.delete_callback(self.mass_name))
        header.addWidget(delete_btn)

        layout.addLayout(header)

        # table of assignments (compact)
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Sous-ensemble", "Valeur"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        table.setColumnWidth(1, 90)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        all_elements = self.mass.get_all_defined_elements()
        table.setRowCount(len(all_elements))

        for row, subset in enumerate(all_elements):
            subset_str = self.mass.frame.format_subset(subset)
            subset_item = QTableWidgetItem(subset_str)
            subset_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 0, subset_item)

            value = self.mass.get_mass(subset)
            value_item = QTableWidgetItem(f"{value:.4f}")
            value_item.setTextAlignment(Qt.AlignCenter)
            if subset in self.explicit_zeros and value == 0:
                value_item.setForeground(QColor("#95a5a6"))
                value_item.setFont(QFont("", italic=True))
            else:
                value_font = QFont()
                value_font.setBold(True)
                value_item.setFont(value_font)
            table.setItem(row, 1, value_item)

        # height calculation to keep cards compact
        row_height = 28
        header_height = 28
        total_height = min(header_height + len(all_elements) * row_height + 8, 220)
        table.setMinimumHeight(total_height)
        table.setMaximumHeight(total_height)

        layout.addWidget(table)
        self.setLayout(layout)


class ResultDisplayWidget(QWidget):
    """Affichage du résultat de fusion : panneau propre et tabulaire"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info header
        self.title_label = QLabel("Résultat de fusion")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #1f2937;")
        layout.addWidget(self.title_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #475569; font-size: 12px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Table
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Sous-ensemble", "Valeur", "Pourcentage"])
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.result_table.setColumnWidth(1, 110)
        self.result_table.setColumnWidth(2, 110)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.result_table)

        # Empty message
        self.empty_label = QLabel("Aucun résultat. Effectuez une fusion pour afficher les résultats.")
        self.empty_label.setStyleSheet("color: #94a3b8; font-style: italic; padding: 18px;")
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label)

        self.clear()

    def display_result(self, result, rule_name, expression=""):
        self.info_label.setText(
            f"<b>Expression:</b> {expression} — <b>Règle:</b> {rule_name}"
        )
        self.title_label.setText(f"Résultat de fusion — {rule_name}")

        all_elements = result.get_all_defined_elements()
        self.result_table.setRowCount(len(all_elements))
        for row, subset in enumerate(all_elements):
            subset_str = result.frame.format_subset(subset)
            subset_item = QTableWidgetItem(subset_str)
            subset_item.setTextAlignment(Qt.AlignCenter)
            subset_font = QFont()
            subset_font.setBold(True)
            subset_item.setFont(subset_font)
            self.result_table.setItem(row, 0, subset_item)

            value = result.get_mass(subset)
            value_item = QTableWidgetItem(f"{value:.6f}")
            value_item.setTextAlignment(Qt.AlignCenter)
            value_font = QFont()
            value_font.setBold(True)
            value_item.setFont(value_font)
            if value == 0:
                value_item.setForeground(QColor("#94a3b8"))
            elif value < 0.1:
                value_item.setForeground(QColor("#d97706"))
            else:
                value_item.setForeground(QColor("#1f5fbf"))
            self.result_table.setItem(row, 1, value_item)

            percent_item = QTableWidgetItem(f"{value*100:.2f} %")
            percent_item.setTextAlignment(Qt.AlignCenter)
            percent_font = QFont()
            percent_font.setItalic(True)
            percent_item.setFont(percent_font)
            self.result_table.setItem(row, 2, percent_item)

        self.empty_label.setVisible(False)
        self.result_table.setVisible(True)
        self.info_label.setVisible(True)

    def clear(self):
        self.info_label.setVisible(False)
        self.result_table.setVisible(False)
        self.empty_label.setVisible(True)
        self.result_table.setRowCount(0)