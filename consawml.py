from typing_extensions import ParamSpecKwargs
from pandas import read_pickle,DataFrame

from sys import stderr

from sklearn.model_selection import train_test_split

from seaborn import heatmap

from keras.layers import Dense,LeakyReLU,BatchNormalization
from keras import Input,Sequential
from keras.metrics import Metric,Precision
from keras.backend import epsilon
from keras.losses import Loss

from tensorflow import reduce_sum,cast,float64,float32,int8,GradientTape,maximum,cast,\
                       logical_and,logical_not,sqrt,shape,multiply,divide,int64
from tensorflow import bool as tfbool,print as tfprint,abs as tfabs
from tensorflow.math import greater_equal
from tensorflow.random import set_seed

from numpy.random import default_rng

from multiprocessing.pool import ThreadPool

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks,\
                                    EditedNearestNeighbours
from imblearn.combine import SMOTETomek

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


eps=epsilon()


def get_tp_tn_fp_fn(y_pred,y_true,data_type=float32):
  if (data_type!=float32 and data_type!=float64):# and y_pred.dtype!=tfbool:
    y_pred_c=cast(y_pred,float32)>.5
    y_true_c=cast(y_true,float32)>.5
  else:
    y_pred_c=y_pred
    y_true_c=y_true
  y_p=cast(y_pred_c,data_type)
  y_t=cast(y_true_c,data_type)
  tp=reduce_sum(y_t*y_p)
  tn=reduce_sum((1-y_t)*(1-y_p))
  fp=reduce_sum((1-y_t)*y_p)
  #fn=reduce_sum(y_true*(1-y_pred))
  #fn=cast(shape(y_true)[0],float32)-(tp+tn+fp)
  return tp,tn,fp,(cast(shape(y_true)[0],data_type)-(tp+tn+fp))

def weigher(a,b):
  a=cast(a,float32)
  b=cast(b,float32)
  return a/(a+b+eps)

def precision(tp,fp):
  return weigher(tp,fp)

def recall(tp,fn):
  return weigher(tp,fn)

def precision_metric(y_pred,y_true):
  tp,_,fp,__=get_tp_tn_fp_fn(y_pred,y_true)
  return weigher(tp,fp)

def recall_metric(y_pred,y_true):
  tp,_,__,fn=get_tp_tn_fp_fn(y_pred,y_true)
  return weigher(tp,fn)

def binary_precision_metric(y_pred,y_true):
  tp,_,fp,__=get_tp_tn_fp_fn(y_pred,y_true,data_type=int64)
  return weigher(tp,fp)

def binary_recall_metric(y_pred,y_true):
  tp,_,__,fn=get_tp_tn_fp_fn(y_pred,y_true,data_type=int64)
  return weigher(tp,fn)

def mk_F_beta(b=1):
  '''
  Provides an f_beta loss function for fixed beta.

  Parameters:
    beta: The parameter to be fixed

  Returns:
    f_beta: The loss function
  '''
  def f_b(y_pred,y_true):
    tp,tn,fp,fn=get_tp_tn_fp_fn(y_pred,y_true)
    p = weigher(tp,fp)
    r = weigher(tp,fn)
    loss=(1 + b ** 2) * (p * r)/((b ** 2 * p) + r + epsilon())
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
    final_loss = - alpha*mcc + (1-alpha)*(penalty_term + fn_penalty)                     # Original formuula

    return final_loss

def evaluate_scheme(scheme,X_train,X_test,y_train,y_test,seed,metrics,epochs,
                    batch_size,l1_size,l2_size,verbose):
  #try:
  resampler=(lambda a,b:(a.copy(),b.copy())) if scheme[1] is None else scheme[1]

  X_sel,y_sel=resampler(X_train,y_train)
  m=mk_two_layer_perceptron(X_sel,scheme[0],seed,metrics=metrics,
                            l1_size=l1_size,l2_size=l2_size)

  m.fit(X_sel,y_sel,epochs=epochs,batch_size=batch_size,verbose=verbose)
  r=list(m.evaluate(X_test,y_test,batch_size=X_test.shape[0]))
  loss_name=str(scheme[0])
  resampler_name=str(scheme[1])
  ret=[loss_name,resampler_name]+r

  if verbose:
    print('Task done after',epochs,' epochs!','Results:',ret,file=stderr)

  return ret

def evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,
                     p=None,epochs=200,batch_size=32,verbose=0,l1_size=64,l2_size=32,
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
  res_col_names=['loss function','resampling scheme','loss']+\
                [str(t) for t in metrics]

  def bench_scheme(s):
    return evaluate_scheme(s,X_train,X_test,y_train,y_test,seed,metrics,
                           epochs,batch_size,l1_size,l2_size,verbose)

  results=p.map(bench_scheme,schemes)
  return DataFrame(results,columns=res_col_names)


def undersample_positive(X,y,seed,p):
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


fh=mk_F_beta(.5)
f1=mk_F_beta(1)
f2=mk_F_beta(2)
f3=mk_F_beta(3)
f4=mk_F_beta(4)
available_losses={'fh':fh,'f1':f1,'f2':f2,'f3':f3,'f4':f4,'binary_crossentropy':'binary_crossentropy',
                  'mean_squared_logarithmic_error':'mean_squared_logarithmic_error',
                  'kl_divergence':'kl_divergence','mean_squared_error':'mean_squared_error',
                  'mcc_fixed_p_fn2':MCCWithPenaltyAndFixedFN_v2(),
                  'mcc_fixed_p_fn3':MCCWithPenaltyAndFixedFN_v3()}

available_resampling_algorithms={'None':None,'SMOTETomek':SMOTETomek().fit_resample,
                                 'random_oversampler':RandomOverSampler().fit_resample,
                                 'SMOTE':SMOTE().fit_resample,'ADASYN':ADASYN().fit_resample,
                                 'random_undersampler':RandomUnderSampler().fit_resample,
                                 'near_miss':NearMiss().fit_resample,'Tomek_links':TomekLinks().fit_resample,
                                 'edited_nearest_neighbours':EditedNearestNeighbours().fit_resample}
