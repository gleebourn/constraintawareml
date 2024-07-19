from pandas import read_pickle

from sklearn.model_selection import train_test_split

from seaborn import heatmap

from keras.layers import Dense,LeakyReLU,BatchNormalization
from keras import Input,Sequential
from keras.metrics import Metric
from keras.backend import epsilon

from tensorflow import reduce_sum,cast,float32
from tensorflow.math import greater_equal

def preproc_bin_class(df,seed,label_col='label',label_class='Malicious',
                      numeric_only=True,test_size=.3,
                      non_feature_cols=['id','ini_timestamp','label','detailed_label']):
  if isinstance(df,str):
    df = read_pickle(df)

  X = df.drop(non_feature_cols+[label_col],axis=1)
  if numeric_only:
    X = X.select_dtypes(include='number')
  else:
    raise NotImplementedError('Only dealing with numerical types for now')
  y = df[label_col]==label_class

  #Returns X_train, X_test, y_train, y_test
  return train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)
  
class FbetaMetric(Metric):
  def __init__(self, beta=1, threshold=0.5, **kwargs):
    super(FbetaMetric, self).__init__(**kwargs)
    self.beta = beta
    self.threshold = threshold
    self.tp = self.add_weight(name='true_positives', initializer='zeros')
    self.fp = self.add_weight(name='false_positives', initializer='zeros')
    self.fn = self.add_weight(name='false_negatives', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred_binary = cast(greater_equal(y_pred, self.threshold),float32)
    #self.tp.assign_add(reduce_sum(cast(y_true,float32) * y_pred_binary))
    y_true_float=cast(y_true,float32)
    self.tp.assign_add(reduce_sum(y_true_float * y_pred_binary))
    self.fp.assign_add(reduce_sum((1 - y_true_float) * y_pred_binary))
    self.fn.assign_add(reduce_sum(y_true_float * (1 - y_pred_binary)))

  def result(self):
    precision = self.tp / (self.tp + self.fp + epsilon())
    recall = self.tp / (self.tp + self.fn + epsilon())
    fbeta = (1 + self.beta ** 2) * (precision * recall)/\
           ((self.beta ** 2 * precision) + recall + epsilon())
    return fbeta


def mk_two_layer_perceptron(df,loss,l1_size=128,l2_size=32,optimizer='adam',
                            #metrics=['precision', 'recall'],#, 'fscore'],
                            metrics=['accuracy','binary_accuracy',
                                     FbetaMetric()],#, 'fscore'],
                            activation=LeakyReLU(alpha=0.01)):

  m=Sequential([Input(shape=df.columns.shape),#,activation=activation),
                BatchNormalization(),Dense(l1_size,activation=activation),
                BatchNormalization(),Dense(l2_size,activation=activation),
                #Just use sigmoid on last step,nicer
                Dense(1,activation='sigmoid')])
  m.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return m

def fit_model(m,training_features,training_labels,
              epochs=200,batch_size=32):
  m.fit(training_features,training_labels,
        epochs=epochs,batch_size=batch_size)

def evaluate_model(m,testing_features,testing_labels):
  m.evaluate(testing_features,testing_labels)
