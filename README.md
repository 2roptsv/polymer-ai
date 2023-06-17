# polymer-ai
Project aimed to use ML tools to predict chemical properties of polymers

### Installation

#### Requirements
* **Python 3** 
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [mol2vec](https://github.com/samoturk/mol2vec/)
* [XGBoost](https://xgboost.ai/)

To install everything run

`python3 -m pip install -r requirements.txt`


### Usage
Main usage - as a desktop application. Run `python3 application.py`

Default model is trained and evaluated with scripts:
* train model: `python3 scripts/train.py`
* evaluate model: `python3 scripts/predict.py`
