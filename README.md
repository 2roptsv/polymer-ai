# polymer-ai
Project aimed to use ML tools to predict chemical properties of polymers

### Installation

#### Requirements
* **Python 3** 
* [Qt5](https://doc.qt.io/qt-5)
* [PyQt5](https://doc.qt.io/qt-5)
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [mol2vec](https://github.com/samoturk/mol2vec/)
* [XGBoost](https://xgboost.ai/)

#### To install everything:
1. [Install Python3] (https://www.python.org/downloads/) (3.9 recommended)
2. If running on:
   1. MacOS arm (m1, m2): install Qt5 (`brew install qt5`), add to PATH, then run `python3 -m pip install pyqt5 --config-settings --confirm-license= --verbose`
   2. Else, just `python3 -m pip isntall pyqt5`
3. Run `python3 -m pip install -r requirements.txt`


### Usage
Main usage - as a desktop application. Run `python3 application.py`

Default model is trained and evaluated with scripts:
* train model: `python3 train.py`
* evaluate model: `python3 predict.py`

**NEW**: experimental NN model trained in PolyInfo dataset (see notebooks/polyinfo.ipynb)
