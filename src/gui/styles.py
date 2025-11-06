"""
Styles pour l'interface graphique
"""

MAIN_STYLE = """
QMainWindow {
    background-color: #ecf0f1;
}

QGroupBox {
    font-weight: bold;
    font-size: 13px;
    border: 2px solid #3498db;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 15px;
    background-color: white;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 8px;
    color: #2c3e50;
}

QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 18px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 12px;
    min-width: 90px;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:pressed {
    background-color: #21618c;
}

QPushButton:disabled {
    background-color: #bdc3c7;
}

QPushButton#addButton {
    background-color: #27ae60;
}

QPushButton#addButton:hover {
    background-color: #229954;
}

QPushButton#deleteButton {
    background-color: #e74c3c;
}

QPushButton#deleteButton:hover {
    background-color: #c0392b;
}

QPushButton#fusionButton {
    background-color: #9b59b6;
    font-size: 15px;
    padding: 12px 24px;
}

QPushButton#fusionButton:hover {
    background-color: #8e44ad;
}

QPushButton#exportButton {
    background-color: #f39c12;
}

QPushButton#exportButton:hover {
    background-color: #e67e22;
}

QLineEdit {
    padding: 8px;
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    background-color: white;
    font-size: 12px;
    color: #2c3e50;
}

QLineEdit:focus {
    border: 2px solid #3498db;
}

/* Style pour QComboBox */
QComboBox {
    padding: 8px;
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    background-color: white;
    font-size: 12px;
    color: #2c3e50;
    font-weight: bold;
}

QComboBox:focus {
    border: 2px solid #3498db;
}

QComboBox:hover {
    border: 2px solid #3498db;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #2c3e50;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: white;
    border: 2px solid #3498db;
    selection-background-color: #3498db;
    selection-color: white;
    color: #2c3e50;
    font-weight: bold;
}

QComboBox QAbstractItemView::item {
    min-height: 30px;
    padding: 5px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #d6eaf8;
    color: #2c3e50;
}

QComboBox QAbstractItemView::item:selected {
    background-color: #3498db;
    color: white;
}

/* Style amélioré pour QSpinBox avec flèches triangulaires et plus large */
QSpinBox {
    padding: 8px 35px 8px 10px;
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    background-color: white;
    font-size: 14px;
    font-weight: bold;
    color: #2c3e50;
    min-width: 100px;
}

QSpinBox:focus {
    border: 2px solid #3498db;
}

QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 30px;
    height: 17px;
    border-left: 1px solid #bdc3c7;
    border-bottom: 1px solid #bdc3c7;
    border-top-right-radius: 3px;
    background-color: #ecf0f1;
}

QSpinBox::up-button:hover {
    background-color: #3498db;
}

QSpinBox::up-button:pressed {
    background-color: #2980b9;
}

QSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-bottom: 9px solid #2c3e50;
    margin: 0 auto;
}

QSpinBox::up-button:hover QSpinBox::up-arrow,
QSpinBox::up-arrow:hover {
    border-bottom-color: white;
}

QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 30px;
    height: 17px;
    border-left: 1px solid #bdc3c7;
    border-top: 1px solid #bdc3c7;
    border-bottom-right-radius: 3px;
    background-color: #ecf0f1;
}

QSpinBox::down-button:hover {
    background-color: #3498db;
}

QSpinBox::down-button:pressed {
    background-color: #2980b9;
}

QSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 9px solid #2c3e50;
    margin: 0 auto;
}

QSpinBox::down-button:hover QSpinBox::down-arrow,
QSpinBox::down-arrow:hover {
    border-top-color: white;
}

QTextEdit {
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    background-color: #fdfefe;
    padding: 10px;
    font-family: 'Courier New', monospace;
    color: #2c3e50;
}

QLabel {
    color: #2c3e50;
    font-size: 12px;
}

QTableWidget {
    border: 2px solid #d5dbdb;
    border-radius: 5px;
    background-color: white;
    gridline-color: #e8f0f5;
    font-size: 12px;
    alternate-background-color: #f8fbfd;
}

QTableWidget::item {
    padding: 8px;
    color: #2c3e50;
}

QTableWidget::item:selected {
    background-color: #3498db;
    color: white;
}

QTableWidget::item:hover {
    background-color: #d6eaf8;
}

QHeaderView::section {
    background-color: #34495e;
    color: white;
    padding: 12px;
    border: none;
    font-weight: bold;
    font-size: 12px;
}

QHeaderView::section:hover {
    background-color: #2c3e50;
}

QScrollBar:vertical {
    border: none;
    background: #ecf0f1;
    width: 12px;
    margin: 0;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #95a5a6;
    border-radius: 6px;
    min-height: 25px;
}

QScrollBar::handle:vertical:hover {
    background: #7f8c8d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollArea {
    border: none;
}
"""