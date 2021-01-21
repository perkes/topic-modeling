# Topic Modeling

A tool to perform topic modeling on a group of strings.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- pip
- virtualenv
- A valid configuration file (see below).

To get **pip**: download [get-pip.py](https://bootstrap.pypa.io/get-pip.py).
```
python get-pip.py
```
Pip is now installed!

To get **virutalenv**: 
```
pip install virtualenv
``` 

### Installing

Create a new virtual environment:
```
virtualenv -p python3 venv
```    

Start the virtual environment:
```
source venv/bin/activate
```    
Install requirements with pip:
```
pip install -r requirements.txt
```

### Configuring

#### Configuring the CSV to JSON configuration file

- **input_doc**: Input data file (CSV).
- **output_doc**: Output data file (CSV).
- **input_column**: Text column to be used as data source for topic modeling.
- **output_column**: Output column for the corresponding topic cluster.
- **topics_number**: Maximum number of topics. *Deafult*: 4.
- **topics_n_words**: Maximum number of words to describe a topic. *Deafult*: 3.
- **min_df**: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. Should be a float in range of [0.0, 1.0], the parameter represents a proportion of documents.  *Deafult*: 0.0.
- **max_df**: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold. Should be a float in range of [0.0, 1.0], the parameter represents a proportion of documents.  *Deafult*: 1.0
- **alpha**: LDA alpha parameter. *Deafult*: 1 / topics_number.
- **beta**: LDA beta parameter. *Deafult*: 1 / topics_number.
- **groupby**: name of the column to group by. This will generate a separate model for each group, and topics_number * number of groups topic columns. If left blank, no grouping will take place.
 
### Running

To run the topic modeling script:
```
python topic_modeling.py
```

## Built With

* [Pandas](https://pandas.pydata.org/) - Data manipulation.
* [Sklearn](https://scikit-learn.org/) - Sci-Kit Learn, Machine Learning in Python.

## Authors

* **Jonathan Perkes** - jperkes@alixpartners.com