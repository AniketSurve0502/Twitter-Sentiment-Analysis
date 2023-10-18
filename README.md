# Twitter Sentiment Analysis

An introduction to Natural Language Processing (NLP) using Analytics Vidhya's Twitter Sentiment Analysis Practice Problem.

## Introduction
This project was a learning experience to become familiar with machine learning tools and libraries in Python. The code in this project was not developed by myself, but was instead sourced from [this guide](https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/).

The purpose of the project was to train a model to predict the sentiment of a tweet. That is, the model was developed to determine whether or not a tweet was racist/sexist. 

To complete this task, a Bag-of-words model was created to meaningfully convey the text in a tweet numerically so that a logistic regression model could be used on the data. The final model was able to predict the sentiment of a tweet with an f1 score of 0.53.

## Running The Program

This project was created using Jupyter Notebook and Anaconda. The following libraries were utilized:
```
import re
import pandas as pd
import numpy as np
import string
import nltk
import warning
import matplotlib.pyplot as plot
import seaborn
```


## Acknowledgments

* [Analytics Vidhya Twitter Sentiment Analysis Challenge](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/)
* [Analytics Vidhya Comprehensive Hands on Guide to Twitter Sentiment Analysis](https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/)

[ThetaPrimo.IO]: https://thetaprime.io	"ThetaPrime.IO"

![](thetaprime_shape.png)
