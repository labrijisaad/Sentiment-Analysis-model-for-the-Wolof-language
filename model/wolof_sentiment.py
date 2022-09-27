import pickle
import sys
import pandas as pd


text = " ".join(sys.argv) # getting the argument

# Load the model
SVM_model = pickle.load(open('model/svm_model.pkl', 'rb'))
SVM_vectorizer = pickle.load(open("model/SVM_vectorizer.pk", "rb"))


def predict_sentiment_svm(text):
    serie = pd.Series(text)
    vector = SVM_vectorizer.transform(serie)
    return str(SVM_model.predict(vector)[0])


def main():
    print(predict_sentiment_svm(text))


if __name__ == "__main__":
    try:
        main()
    except:
        print("Something went wrong :(")


















