# polymer-ai
Project aimed to use ML tools to predict chemical properties of polymers

### Installation

#### Requirements
* **Python 3** 
* [Qt5] (https://doc.qt.io/qt-5)
* [PyQt5] (https://doc.qt.io/qt-5)
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [mol2vec](https://github.com/samoturk/mol2vec/)
* [XGBoost](https://xgboost.ai/)

#### To install everything:
1. [Install Python3] (https://www.python.org/downloads/)
2. [Install Qt5] (https://doc.qt.io/qt-5/gettingstarted.html) - *don't forget to add Qt5 qmake to PATH, instructions are in the end of installation*
3. Run `python3 -m pip install -r requirements.txt`


### Usage
Main usage - as a desktop application. Run `python3 application.py`

Default model is trained and evaluated with scripts:
* train model: `python3 scripts/train.py`
* evaluate model: `python3 scripts/predict.py`
