MAIN_STYLE = """
/* Fenêtre principale */
QMainWindow {
  background-color: #f5f7fa;
  color: #1f2937;
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 12px;
}

/* Menu / Toolbar / Status */
QMenuBar {
  background: transparent;
  spacing: 4px;
}
QMenuBar::item {
  padding: 6px 10px;
  color: #1f2937;
}
QMenuBar::item:selected {
  background: #e6f0ff;
  color: #1f5fbf;
}

QToolBar {
  background: transparent;
  padding: 4px;
  spacing: 6px;
  border: none;
}

QStatusBar {
  background: transparent;
  border-top: 1px solid #e6eef9;
  padding: 4px 8px;
  color: #4a5568;
}

/* Group boxes (cards) */
QGroupBox {
  border: 1px solid #e6eef6;
  border-radius: 6px;
  margin-top: 8px;
  padding: 10px;
  background: #ffffff;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 3px;
  color: #2d3748;
  font-weight: 600;
  font-size: 12px;
}

/* Mass card */
QFrame#massCard {
  background: #ffffff;
  border: 1px solid #e6eef6;
  border-radius: 8px;
  padding: 8px;
}
QFrame#massCard:hover {
  border-color: #cbd5e1;
}

/* Buttons */
QPushButton {
  background-color: #1f5fbf;
  color: #ffffff;
  border: none;
  padding: 8px 12px;
  border-radius: 6px;
  font-weight: 600;
  min-height: 32px;
}
QPushButton:hover {
  background-color: #184f9e;
}
QPushButton:pressed {
  background-color: #133f7a;
}
QPushButton#secondary {
  background-color: transparent;
  color: #1f5fbf;
  border: 1px solid transparent;
  font-weight: 600;
}
QPushButton#danger {
  background-color: #e53e3e;
}
QPushButton#deleteButton {
  background: transparent;
  color: #c53030;
  border: 1px solid transparent;
  padding: 4px 8px;
}

/* Flat / small tool buttons */
QPushButton[flat="true"], QPushButton.flat {
  background: transparent;
  border: none;
  color: #4a5568;
  padding: 4px;
}

/* Inputs */
QLineEdit, QComboBox, QSpinBox {
  background: #ffffff;
  border: 1px solid #dbe7f7;
  border-radius: 6px;
  padding: 8px;
  color: #1f2937;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
  border: 1px solid #97c2ff;
  outline: none;
}

/* Table */
QTableWidget {
  background: #ffffff;
  border: 1px solid #e6eef6;
  gridline-color: #f1f5f9;
  alternate-background-color: #fbfdff;
}
QTableWidget::item {
  padding: 6px;
}
QHeaderView::section {
  background: #f1f5f9;
  padding: 8px;
  border: none;
  font-weight: 700;
  color: #1f2937;
}

/* Result table coloring */
QTableWidget::item:selected {
  background: #dbeafe;
  color: #0b254f;
}

/* Scrollbars (subtle) */
QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 0px;
}
QScrollBar::handle:vertical {
  background: #cbd5e1;
  min-height: 20px;
  border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
  background: #9aa9b8;
}

/* Tooltips */
QToolTip {
  background-color: #1f2937;
  color: #ffffff;
  border: none;
  padding: 6px;
  border-radius: 4px;
  opacity: 230;
}
"""