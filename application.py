import os
from typing import Any, Callable, Dict, List
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

from src.model import ModelWrapper
from src.defaults import DEFAULT_INPUTS, TRANSLATION, POLYMER_CLASS


class Window(QMainWindow):
    def _get_on_click_callback(self):
        def callback():
            keyword_inputs, smiles = self._collect_data()
            print(smiles)
            print(keyword_inputs)
            predictions, metrics = self._callback(keyword_inputs, smiles)
            print(predictions, metrics)

            msg = QMessageBox()
            msg.setWindowTitle("Success!")
            lines = ["Predictions: {:.2f}, MAPE {:.2f}".format(predictions[target], metrics[target])
                     for target in predictions]
            msg.setText('\n'.join("Predictions:", *lines))
            x = msg.exec_()

            self.update()
        return callback

    def _collect_data(self):
        keyword_inputs = {k: float(v.text()) for k, v in self.text_boxes.items() if v.text() != ''}
        polymer_class = self.polymer_class_dropbox.currentText()
        keyword_inputs[POLYMER_CLASS] = polymer_class
        smiles = self.smiles_textbox.text()
        return keyword_inputs, smiles

    def _build_ui(self):
        self.setGeometry(200, 200, self._w, self._h)
        self.setWindowTitle("polymer-ai")

        # Label
        self.label = QLabel(self)
        self.label.setText("Enter polymer properties and SMILES")
        self.label.adjustSize()
        self.label.move((self.width() - self.label.width()) // 2, 20 - self.label.height())
        current_occupied_height = 20 + self.label.height()

        # Polymer drop list
        label = QLabel(self)
        label.setText(f"{TRANSLATION[POLYMER_CLASS]}:")
        label.setFixedSize(200, 20)
        label.move(20, current_occupied_height)
        self.polymer_class_dropbox = QComboBox(self)
        self.polymer_class_dropbox.addItems(self._polymer_classes)
        self.polymer_class_dropbox.move(20 + 200, current_occupied_height)
        current_occupied_height += self.polymer_class_dropbox.height() + 20

        # Properties
        textboxes_height = self.height() - current_occupied_height - 300
        textbox_width = 100
        textbox_height = 20
        spacing = (textboxes_height - len(DEFAULT_INPUTS) * textbox_height)
        spacing //= (len(DEFAULT_INPUTS) - 1)
        self.text_boxes = {}
        self.text_field_labels = {}
        for index, column in enumerate(DEFAULT_INPUTS):
            # Label
            label = QLabel(self)
            label.setText(f"{TRANSLATION[column]}:")
            label.setFixedSize(200, textbox_height)
            label.move(20, current_occupied_height + index * (textbox_height + spacing))
            self.text_field_labels[column] = label
            # Textbox
            textbox = QLineEdit(self)
            textbox.move(20 + 200, current_occupied_height + index * (textbox_height + spacing))
            textbox.resize(textbox_width, textbox_height)
            validator = QDoubleValidator()
            validator.setLocale(QLocale("en_US"))
            textbox.setValidator(validator)
            self.text_boxes[column] = textbox
        current_occupied_height += textboxes_height + 40

        # SMILES
        label = QLabel(self)
        label.setText(f"SMILES:")
        label.setFixedSize(100, textbox_height)
        label.move(20, current_occupied_height)
        self.smiles_label = label

        textbox = QLineEdit(self)
        textbox.move(20 + 100, current_occupied_height)
        textbox.resize(300, textbox_height)
        self.smiles_textbox = textbox

        self.main_button = QPushButton(self)
        self.main_button.setText("predict")
        self.main_button.move((self.width() - self.main_button.width()) // 2,
                              self.height() - self.main_button.height() // 2 - 20)
        self.main_button.clicked.connect(self._get_on_click_callback())

    def __init__(
            self,
            callback,
            polymer_classes: List[str],
            width=500,
            height=700
    ):
        super(Window, self).__init__()
        self._callback = callback
        self._polymer_classes = polymer_classes
        self._w = width
        self._h = height
        self._build_ui()


class Application:
    def _model_callback(self, input_kwargs: Dict, smiles: str):
        return self._model(input_kwargs, smiles)

    def __init__(self, model: ModelWrapper):
        self._model = model
        self.app = QApplication(sys.argv)
        self.window = Window(self._model_callback, self._model.get_possible_category_values(POLYMER_CLASS))
        self.window.show()

    def stop(self):
        sys.exit(self.app.exec_())


def run():
    checkpoint_path = Path(os.path.abspath(__file__)).parent / "resources/models/test"
    model = ModelWrapper(checkpoint_path)
    a = Application(model)
    a.stop()


run()
