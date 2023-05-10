import time
import ray
from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tune_sklearn import TuneGridSearchCV

def train_model():
    #get the data
    x, y = fetch_covtype(return_X_y=True)

    #split test and train data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

    #the model: random forest classifier
    model = RandomForestClassifier()
    parameter_grid = {"n_estimators": [50, 100, 125], "max_depth": [5, 10, 15, 20], "ccp_alpha": [0, >
    tune_search = TuneGridSearchCV(model, parameter_grid)

    tune_search.fit(x_train, y_train)

    pred = tune_search.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    print("Accuracy with tuning: ", accuracy)

    print(tune_search.best_params_)


def main():
    start = time.time()
    train_model()
    end = time.time()
    print("time: ", end-start)




if __name__ == '__main__':
        main()
