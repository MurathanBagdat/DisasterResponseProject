# DisasterResponseProject

# Aim
Purpose of this project is to build a ML model that uses Natural Language Processing to classify messages during a natural disaster.

# Structure

Project is consist of 3 parts.
- ETL
- Machine Learning Pipeline
- Web Application

# ETL
In ETL part data_process.py script extracts data from 2 diffrent csv files, merges them and transform them according to our needs.
After that it loads the data in a user specified SQL database.

To run ETL script you need to specify filepaths for csv files and which SQL database you want to write. 
Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

# Machine Learning Pipeline
In ML Pipeline part data that is loaded in ETL proceses is used. Messages are tokenized and converted in to bag of words.
To train a model sklearn pipeline library is used. Best performing classfier is choosen as RandomForestClassifier after some trail and erorr. Appropriate hyperparamater are choosen using grid search. Trained model saved as .pkl file.

# Web Application
Also there is a web application that takes user input and classify it using trained model that is trained in ML pipeline part. 
Here you can see preview of this web app.

![](/Project/webapp.JPG)
