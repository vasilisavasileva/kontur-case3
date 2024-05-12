from typing import Union

from fastapi import FastAPI

from some_ml import *

app = FastAPI()


@app.get("/decisionTree")
def decisionTree_handler():
    score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
    return { "score": score, "report": report, "tree": tree }




@app.get("/neuralNet")
def neuralNet_handler():
    score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
    return { "score": score, "report": report, "clf": clf }



@app.get("/Knn_Classifier")
def Knn_Classifier_handler():
    score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
    return { "score": score, "report": report, "clf": clf }