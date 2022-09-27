# 📈  `Sentiment Analysis model for the Wolof language` 🌍:
<p align="center">
  <img src="https://user-images.githubusercontent.com/74627083/192595184-9c41205f-6198-43cc-9c64-074e7c0609ef.png" />
</p>
  
 - 🎯 In this notebook, I tried to create a **`sentiment analysis model for Wolof language`**. 
- 🤔 To do this, I gathered data already labeled in different datasets extracted from different sources (Quran, Bible, proverbs, twitter): The notebooks I used for web scraping can be found in my [github repository](https://github.com/labrijisaad)).
- ⏭ Next, i used some of scikit learn's most well-known classification models with **`TF-IDF Vectorizer`** to train the final model:
  - **`Logistic Regression`**
  - **`Bernoulli Naive Bayes`**                                                
  - **`Decision Tree`**
  - **`Support vector machine`** 
 
<br>

- Here are **TWO** ways to use the trained model in notebook: (You must before install the requirements)

##### meth 1
> via model and vectorizer import

```py
    import pickle
    import pandas as pd
    import re

    SVM_model = pickle.load(open('SVM_model.pkl', 'rb'))
    SVM_vectorizer = pickle.load(open("SVM_vectorizer.pk","rb"))
    
    LR_model = pickle.load(open('LR_model.pkl', 'rb'))
    LR_vectorizer = pickle.load(open("LR_vectorizer.pk","rb"))

   def process_wolof_text(text): ## text processing
      text = text.lower()
      text = remove_ponctuation(text)
      text = cleaning_numbers(text)
      text = cleaning_stopwords(text)
      text = remove_ponctuation(text)
      text = re.sub("\s\s+", " ", text)
      return text

    def predict_sentiment_svm(text):
       serie = pd.Series(text)
       vector = SVM_vectorizer.transform(serie)
       return str(SVM_model.predict(vector)[0])

    def predict_wolof_with_logistic_regression(text):
       text = process_wolof_text(text)
       text = [text]
       text = LR_vectorizer.transform(text)
       return "POSITIVE" if (LR_model.predict(text)[0] == 1) else "NEGATIVE"

    
    text = "dafa xëm"  # Il est évanoui
    print(predict_wolof_with_logistic_regression(text))
    print(predict_sentiment_svm(text))
    
    >>> NEGATIVE
    >>> NEGATIVE
 ```

##### meth 2
> by calling a script that does all the work for us

```py
    text = "Na nga def ?"
    var = !python model/wolof_sentiment.py $text 
    print(var[-1])
    
    >>> 1
```

- 💪 Model performance: Here are the results obtained after training the model (SVM and LR are the models that performed well) 
                               
                                                          LOGISTIC REGRESSION
                             
                                               precision    recall  f1-score   support        

                                    -1             0.82      0.78      0.80       512         
                                     1             0.77      0.82      0.79       475         
                                accuracy                               0.80       987         
                                macro avg          0.80      0.80      0.80       987         
                                weighted avg       0.80      0.80      0.80       987        
                                          

                                                        SUPPORT VECTOR MACHINE
                                                        
      Positive:  {'precision': 0.8136645962732919, 'recall': 0.7844311377245509, 'f1-score': 0.7987804878048781, 'support': 501}
      Negative:  {'precision': 0.7857142857142857, 'recall': 0.8148148148148148, 'f1-score': 0.7999999999999999, 'support': 486}

- 📫 Feel free to contact me if anything is wrong or if anything needs to be changed 😎!  **labrijisaad@gmail.com**

<a href="https://colab.research.google.com/github/labrijisaad/Sentiment-Analysis-model-for-the-Wolof-language" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

> - 🙌 Notebook made by [@labriji_saad](https://github.com/labrijisaad)
> - 🔗 Linledin [@labriji_saad](https://www.linkedin.com/in/labrijisaad/)
