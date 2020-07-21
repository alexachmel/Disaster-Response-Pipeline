# Disaster Response Pipeline

For this project I applied data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Installations:
In this project Python 3.x and the following Python libraries were installed:<br>
Pandas https://pandas.pydata.org/<br>
Numpy https://numpy.org/<br>
Plotly https://plotly.com/<br>
Pickle https://docs.python.org/3/library/pickle.html<br>
Json https://www.json.org/<br>
Sqlalchemy https://www.sqlalchemy.org/<br>
Flask https://flask.palletsprojects.com/en/1.1.x/<br>
Re https://docs.python.org/3/library/re.html<br>
NLTK https://www.nltk.org/<br>
Sklearn https://scikit-learn.org/stable/<br>


### There are 3 Components in this project:
1. __ETL Pipeline__<br>
In a Python script, process_data.py, there's a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
2. __ML Pipeline__<br>
In a Python script, train_classifier.py, there's a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file
3. __Flask Web App__<br>
- The flask web app including data visualizations using Plotly in the web app.


### An explanation of the files
1. ETL Pipeline Preparation.ipynb / ETL Pipeline Preparation.pdf # cleans data and stores in database<br>
2. ML Pipeline Preparation.ipynb / ML Pipeline Preparation.pdf # trains classifier and saves<br>
3. app<br>
3.1 template<br>
3.1.1 master.html  # main page of web app<br>
3.1.2 go.html  # classification result page of web app<br>
3.2 run.py  # Flask file that runs app<br>
4. data<br>
4.1 disaster_categories.csv  # data to process <br>
4.2 disaster_messages.csv  # data to process<br>
4.3 process_data.py<br>
4.4 InsertDatabaseName.db   # database to save clean data to<br>
5. models<br>
5.1 train_classifier.py<br>
5.2 classifier.pkl  # saved model <br>


### How to run the Python scripts and web app
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

