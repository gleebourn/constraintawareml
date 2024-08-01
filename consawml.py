from typing_extensions import ParamSpecKwargs
from pandas import read_pickle

from sys import exc_info

from sklearn.model_selection import train_test_split

from seaborn import heatmap

from keras.layers import Dense,LeakyReLU,BatchNormalization
from keras import Input,Sequential
from keras.metrics import Metric,Precision
from keras.backend import epsilon
from keras.losses import Loss

from tensorflow import reduce_sum,cast,float32,int8,GradientTape,maximum,cast,\
                       logical_and,logical_not,sqrt,shape,multiply,divide
from tensorflow import bool as tfbool,print as tfprint,abs as tfabs
from tensorflow.math import greater_equal
from tensorflow.random import set_seed
#rom tensorflow.debugging import disable_traceback_filtering
#disable_traceback_filtering()

from numpy.random import default_rng

from multiprocessing.pool import ThreadPool

def preproc_bin_class(df,seed,label_col='label',label_class='Malicious',
                      numeric_only=True,test_size=.3,
                      non_feature_cols=['id','ini_timestamp','label','detailed_label']):
  '''
  Pick out relevant columns from a dataframe and split it into training and test data

  Parameters:
    df: Dataframe or is a string treated as the location of a pickled DataFrame
    seed: Random seed for train-test random splitting purposes
    label_col: The name of the column corresponding to the classification problem
    label_class: The value in label_col corresponding to the class of interest
    numeric_only: Ignore categoricL features
    test_size: The proportion of the data to use for testing
    non_feature_cols: List of columns to not use as features
  
  Returns:
    X_train: Features for training
    X_test: Features for testing
    y_train: Labels for training
    y_test: Labels for testing
  '''

  set_seed(seed)

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


def mk_two_layer_perceptron(X,loss,seed,l1_size=128,l2_size=32,optimizer='adam',
                            metrics=['accuracy','binary_accuracy',
                                     FbetaMetric()],
                            activation=LeakyReLU(alpha=0.01)):
  '''
  Generate the two layer perceptron that we train using loss functions of interest

  Parameters:
    X: A DataFrame of features - used to figure out the input layer size
    loss: The loss function to train the model on
    seed: Random seed for the stochastic training algorithm to use

  Returns:
    m: A model to be trained by fit_model
  '''

  set_seed(seed)

  m=Sequential([Input(shape=X.columns.shape),#,activation=activation),
                BatchNormalization(),Dense(l1_size,activation=activation),
                BatchNormalization(),Dense(l2_size,activation=activation),
                #Just use sigmoid on last step,nicer
                Dense(1,activation='sigmoid')])
  m.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return m


def evaluate_model(m,testing_features,testing_labels):
  '''
  Test and score a trained model using the test data

  Parameters:
    m: The model to be trained
    testing_features: The features to be queried
    testing_labels: The correct classifications of the features
  '''
  m.evaluate(testing_features,testing_labels)

def evaluate_metric(df,seed,loss='binary_crossentropy',
                    epochs=200,batch_size=32):
  X_train,X_test,y_train,y_test = preproc_bin_class(df,seed)
  m=mk_two_layer_perceptron(X_train,loss)
  m.fit(X_train,y_train,
        epochs=epochs,batch_size=batch_size)
  m.evaluate(X_test,y_test)


class MatthewsCorrelationCoefficient(Metric):
    def __init__(self, name='matthews_correlation', **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = cast(y_true, tfbool)
        y_pred = cast(y_pred > 0.5, tfbool)

        tp = reduce_sum(cast(logical_and(y_true, y_pred), float32))
        tn = reduce_sum(cast(logical_and(logical_not(y_true),
                                   logical_not(y_pred)),
                           float32))
        fp = reduce_sum(cast(logical_and(logical_not(y_true), y_pred),
                           float32))
        fn = reduce_sum(cast(logical_and(y_true, logical_not(y_pred)),
                           float32))

        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        mcc = (self.true_positives * self.true_negatives -
               self.false_positives * self.false_negatives) / (
              sqrt((self.true_positives + self.false_positives) *
                      (self.true_positives + self.false_negatives) *
                      (self.true_negatives + self.false_positives) *
                      (self.true_negatives + self.false_negatives)) +
                      epsilon())
        return mcc

    def reset_states(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

def mk_F_beta(b=1):
  '''
  Provides an f_beta loss function for fixed beta.

  Parameters:
    beta: The parameter to be fixed

  Returns:
    f_beta: The loss function
  '''
  def f_b(y_pred,y_true):
    y_true=cast(y_true,float32)
    y_pred=cast(y_pred,float32)
    true_positives=reduce_sum(y_true*y_pred)
    true_negatives=reduce_sum((1-y_true)*(1-y_pred))
    false_positives=reduce_sum((1-y_true)*y_pred)
    #false_negatives=reduce_sum(y_true*(1-y_pred))
    false_negatives=cast(shape(y_true)[0],float32)-\
                    (true_positives+true_negatives+false_positives)
    precision = true_positives / (true_positives + false_positives + epsilon())
    recall = true_positives / (true_positives+false_negatives + epsilon())
    loss=(1 + b ** 2) * (precision * recall)/\
         ((b ** 2 * precision) + recall + epsilon())
    return 1-loss
  
  f_b.__qualname__='f'+str(b)
  f_b.__name__='f'+str(b)

  return f_b

f_b_1=mk_F_beta(1)
f_b_2=mk_F_beta(2)
f_b_3=mk_F_beta(3)

class MCCWithPenaltyAndFixedFN_v2(Loss):
  def __init__(self, fp_penalty_weight=0.68, fixed_fn_rate=0.2, tradeoff_weight=1.0):
    super(MCCWithPenaltyAndFixedFN_v2, self).__init__()
    self.fp_penalty_weight = fp_penalty_weight
    self.fixed_fn_rate = fixed_fn_rate
    self.tradeoff_weight = tradeoff_weight

  def call(self, y_true, y_pred):
    targets = cast(y_true, dtype=float32)
    inputs = cast(y_pred, dtype=float32)
    tp = reduce_sum(multiply(inputs, targets))
    tn = reduce_sum(multiply((1 - inputs), (1 - targets)))
    fp = reduce_sum(multiply(inputs, (1 - targets)))
    fn = reduce_sum(multiply((1 - inputs), targets))
    epsilon = 1e-7

    # Calculate the fixed number of false negatives
    fixed_fn = self.fixed_fn_rate * reduce_sum(targets)

    # Introduce a penalty term for false positives
    penalty_term = self.tradeoff_weight * self.fp_penalty_weight * fp

    numerator = tp * tn - fp * fn
    denominator = sqrt((tp + fp + epsilon) * (tp + fn + epsilon) * (tn + fp + epsilon) * (tn + fn + epsilon))
    mcc = divide(numerator, denominator)

    # Add the penalty term to the MCC

    penalty_term=0
    mcc_with_penalty = mcc - penalty_term

    # Add a penalty term to keep false negatives at a fixed rate
    fn_penalty = maximum(0.0, fn - fixed_fn)
    fn_penalty=0
    # Adjust the final loss with the false negative penalty
    final_loss = -mcc_with_penalty + fn_penalty


    return final_loss

class MCCWithPenaltyAndFixedFN_v3(Loss):
  def __init__(self, fp_penalty_weight=0.8, fixed_fn_rate=0.2, tradeoff_weight=1.0):
    super(MCCWithPenaltyAndFixedFN_v3, self).__init__()
    self.fp_penalty_weight = fp_penalty_weight
    self.fixed_fn_rate = fixed_fn_rate
    self.tradeoff_weight = tradeoff_weight

  def call(self, y_true, y_pred):
    targets =cast(y_true, dtype=float32)
    inputs =cast(y_pred, dtype=float32)

    tp = reduce_sum(multiply(inputs, targets))
    tn = reduce_sum(multiply((1 - inputs), (1 - targets)))
    fp = reduce_sum(multiply(inputs, (1 - targets)))
    fn = reduce_sum(multiply((1 - inputs), targets))
    epsilon = 1e-7


    fixed_fn = self.fixed_fn_rate * fn
    fn_penalty = maximum(0.0, fn - fixed_fn)

    # Introduce a penalty term for false positives
    penalty_term = self.tradeoff_weight * self.fp_penalty_weight * fp

    numerator = tp
    denominator = sqrt((tp + fp + epsilon) * (tp + fn + epsilon))
    mcc = divide(numerator, denominator)

    # Scale each penalty term to be between -1 and 1
    max_abs_penalty = maximum(tfabs(penalty_term),tfabs(fn_penalty))
    scaled_penalty_term = penalty_term / max_abs_penalty
    scaled_fn_penalty = fn_penalty / max_abs_penalty

    #return scaled_penalty_term, scaled_fn_penalty

    alpha = 0.6

    # Adjust the final loss with the MCC and penalty terms
    #final_loss = scaled_penalty_term + scaled_fn_penalty                                 # Woks fine No need for the MCC term
    final_loss = - alpha*mcc + (1-alpha)*(penalty_term + fn_penalty)                     # Original formuula
    #final_loss = - alpha*mcc + (1-alpha)*(scaled_penalty_term + scaled_fn_penalty)        # Sclaed formuula
    #final_loss = -mcc + (scaled_penalty_term + scaled_fn_penalty)                        # Sclaed formuula without weight  No so good-Danger =25% FA=1.2%

    #tf.print("mcc value:", mcc)
    '''
    tfprint("tp :", tp)
    tfprint("fp :", fp)

    tfprint("fn :", fn)
    tfprint("fixed_fn :", fixed_fn)

    tfprint("fn_penalty :", fn_penalty)
    tfprint("(1-alpha)*(penalty_term + fn_penalty) value:", (1-alpha)*(penalty_term + fn_penalty))

    print("Shape of the mcc:", mcc.shape)
    print("Shape of the final_loss:", final_loss.shape)

    # Check if the variable is a tensor
    if tf.is_tensor(mcc):
        tf.print("mcc: ", mcc)

    else:
        print("Tmcc =", mcc)
    print("mcc=",mcc)
    '''

    return final_loss

def evaluate_scheme(scheme,X_train,X_test,y_train,y_test,
                    seed,metrics,epochs,batch_size):
  try:
    resampler=(lambda a,b:(a.copy(),b.copy())) if scheme[1] is None else scheme[1]

    X_sel,y_sel=resampler(X_train,y_train)
    m=mk_two_layer_perceptron(X_sel,scheme[0],seed,metrics=metrics)

    m.fit(X_sel,y_sel,epochs=epochs,batch_size=batch_size)
    r=m.evaluate(X_test,y_test,batch_size=X_test.shape[0])
    n=m.metrics
  except Exception as e:
    print()
    print('===========================================================')
    tfprint('Error encountered!',e)
    exinf=exc_info()
    print('Line',exinf[2].tb_lineno)
    print('===========================================================')
    print()
    r=[]
    n=[]
  return r,n

def evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,
                     p=None,epochs=200,batch_size=32,
                     metrics=['accuracy','binary_accuracy',f_b_1,f_b_2,f_b_3]):
  '''
  Evaluate various approaches to learning unbalanced data

  Parameters:
    schemes: A list of tuples a,b where a is a loss function, and
             b:(X_train,y_train)->(X_sel,y_sel) is a resampled version of the
             training data.  If b is None there is no resampling
    X_train: Features for training
    X_test: Features for testing
    y_train: Labels for training
    y_test: Labels for testing
    seed: Random seed for selection and training algorithms to use
  
  Returns:
    results: A list of tuples corresponding to the metrics for each scheme
  '''
  set_seed(seed)
  results=[]
  p=ThreadPool()#len(schemes))
  #Parallelised evaluation of schemes
  return p.map(lambda s:evaluate_scheme(s,X_train,X_test,y_train,y_test,seed,
                                        metrics,epochs,batch_size),schemes)

def undersample_positive(X,y,seed,p=.5):
  choice=default_rng(seed=seed).choice
  '''
  Undersamples the data X,y so that the proportion in the
  positive class is p.
  Parameters:
    X: Features to undersample
    y: Labels to undersample
    seed: Seed for the random choice of positive rows
    p: Proportion of the output data that is positively labelled
  
  Returns:
    results: A list of tuples corresponding to the metrics for each scheme
  
  '''
  n_rows=len(y)
  n_positive=y.sum()
  n_negative=n_rows-n_positive
  new_n_positive=int((p*n_negative)/(1-p))
  num_to_drop=n_positive-new_n_positive
  if num_to_drop<0:
    raise ValueError('Already too few positive rows!')
  positive_indices=y.index[y]
  rows_to_drop=choice(positive_indices,num_to_drop,
                      replace=False)

  return X.drop(rows_to_drop),y.drop(rows_to_drop)

