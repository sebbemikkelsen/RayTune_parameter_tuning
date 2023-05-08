import ray
from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    #get the data
    x, y = fetch_covtype(return_X_y=True)

    #split test and train data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

    #the model: random forest classifier
    model = RandomForestClassifier()

    #train model
    model.fit(x_train, y_train)

    #make predictions
    prediction = model.predict(x_test)

    #check accuracy
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy with default parameters: ", accuracy)


def main():
    train_model()




if __name__ == '__main__':
	main()
