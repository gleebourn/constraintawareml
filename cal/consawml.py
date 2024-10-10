from sys import stderr
from time import thread_time
from multiprocessing.pool import ThreadPool

from numpy.random import default_rng

from pandas import read_pickle,DataFrame

from sklearn.model_selection import train_test_split

from keras.losses import Loss
from keras import Input,Sequential
from keras.layers import Dense,LeakyReLU,BatchNormalization
from keras.backend import epsilon

from tensorflow import reduce_sum,divide,cast,float64,float32,cast,shape,maximum,sqrt,\
                       multiply,abs as tfabs
from tensorflow.math import log
from tensorflow.random import set_seed


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
  y = cast(df[label_col]==label_class,float64)

  #Returns X_train, X_test, y_train, y_test
  return train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)

#Algebraic definitions of false/true positives/negatives
#Less efficient to calculate if we know y_t is going to be one-hot encoded.
flo1=cast(1,float64)
def cts_fp(y_t,y_p): return reduce_sum((flo1-cast(y_t,float64))*cast(y_p,float64))
def cts_fn(y_t,y_p): return reduce_sum((flo1-cast(y_p,float64))*cast(y_t,float64))
def cts_tp(y_t,y_p): return reduce_sum(cast(y_t,float64)*cast(y_p,float64))
def cts_tn(y_t,y_p): return reduce_sum((flo1-cast(y_t,float64))*(flo1-cast(y_p,float64)))

#Algebraic definitions of false/true positives/negatives
def bin_fp(y_t,y_p): return cts_fp(y_t,y_p>.5)
def bin_fn(y_t,y_p): return cts_fn(y_t,y_p>.5)
def bin_tp(y_t,y_p): return cts_tp(y_t,y_p>.5)
def bin_tn(y_t,y_p): return cts_tn(y_t,y_p>.5)
  
def mk_two_layer_perceptron(X,loss,seed,l1_size=128,l2_size=32,optimizer='adam',
                            metrics=['accuracy','binary_accuracy'],
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
                Dense(l1_size,activation=activation),
                #BatchNormalization(),Dense(l1_size,activation=activation),
                BatchNormalization(),Dense(l2_size,activation=activation),
                #Just use sigmoid on last step,nicer
                Dense(1,activation='sigmoid')])
                #Dense(1,activation=activation)])
  m.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return m

def evaluate_metric(df,seed,loss='binary_crossentropy',
                    epochs=200,batch_size=32):
  X_train,X_test,y_train,y_test = preproc_bin_class(df,seed)
  m=mk_two_layer_perceptron(X_train,loss)
  m.fit(X_train,y_train,
        epochs=epochs,batch_size=batch_size)
  m.evaluate(X_test,y_test)

def cts_precision(y_true,y_pred):
  y_true_cts=cast(y_true,float64) #!?!?!
  y_pred_cts=cast(y_true,float64)
  tp=cts_tp(y_true_cts,y_pred_cts)
  fp=cts_fp(y_true_cts,y_pred_cts)
  return tp/(tp+fp+epsilon())

def cts_recall(y_true,y_pred):#,data_type=float64):
  y_true_cts=cast(y_true,float64) #!?!?!
  y_pred_cts=cast(y_true,float64)
  tp=cts_tp(y_true_cts,y_pred_cts)
  fn=cts_fn(y_true_cts,y_pred_cts)
  return tp/(tp+fn+epsilon())

def binary_precision(y_true,y_pred):
  return cts_precision(y_true,cast(y_pred>.5,float64))

def binary_recall(y_true,y_pred):
  return cts_recall(y_true,cast(y_pred>.5,float64))

def mk_F_beta(b=1):
  '''
  Provides an f_beta loss function for fixed beta.

  Parameters:
    b: The parameter to be fixed

  Returns:
    f_b: The loss function
  '''
  b2=b**2
  def f_b(y_true,y_pred):
    tp=cts_tp(y_true,y_pred)
    fp=cts_fp(y_true,y_pred)
    fn=cts_fn(y_true,y_pred)
    a=fp+b2*fn
    b=(1+b2)*tp
    return a/(a+b)
  
  f_b.__qualname__='f'+str(b)
  f_b.__name__='f'+str(b)

  return f_b

#def mk_g_beta(b=1):
def mk_gp_beta(b=1):
  '''
  Provides a simplified F_beta metric

  Parameters:
    b: The parameter to be fixed

  Returns:
    fr_b: The loss function
  '''
  b2=b**2
  def fr_b(y_t,y_p):
    fp=cts_fp(y_t,y_p)
    fn=cts_fn(y_t,y_p)
    tp=cts_tp(y_t,y_p)
    return (fp+b2*fn)/tp
  
  fr_b.__qualname__='g'+str(b)
  fr_b.__name__='g'+str(b)

  return fr_b

def mk_gn_beta(b=1):
  '''
  Provides a simplified F_beta metric

  Parameters:
    b: The parameter to be fixed

  Returns:
    fr_b: The loss function
  '''
  b2=b**2
  def fr_b(y_t,y_p):
    fp=cts_fp(y_t,y_p)
    fn=cts_fn(y_t,y_p)
    tn=cts_tp(y_t,y_p)
    return (fp+b2*fn)/tn
  
  fr_b.__qualname__='g'+str(b)
  fr_b.__name__='g'+str(b)

  return fr_b

def mk_rebalanced_ls(b=1):
  '''
  Provides a rebalanced least squares loss

  Parameters:
    b: The parameter to be fixed

  Returns:
    fr_b: The loss function
  '''
  b2=b**2
  def fr_b(y_t,y_p):
    fp=cts_fp(y_t,y_p)
    fn=cts_fn(y_t,y_p)
    return fp+b2*fn
  
  fr_b.__qualname__='fr'+str(b)
  fr_b.__name__='fr'+str(b)

  return fr_b

def mk_rebalanced_lq(b=1):
  '''
  Provides a rebalanced least quartic loss

  Parameters:
    b: The parameter to be fixed

  Returns:
    fq_b: The loss function
  '''
  b2=b**2
  def fq_b(y_t,y_p):
    fp=cts_fp(y_t,y_p)
    fn=cts_fn(y_t,y_p)
    return fp+b2*fn
  
  fq_b.__qualname__='fq'+str(b)
  fq_b.__name__='fq'+str(b)

  return fq_b

def mk_log_F_beta(b=1):
  '''
  Provides log f_beta loss function for fixed beta.

  Parameters:
    beta: The parameter to be fixed

  Returns:
    log_f_beta: The loss function
  '''
  b2=b**2
  def f_b(y_true,y_pred): return log(f_beta(y_true,y_pred))
  
  f_b.__qualname__='lf'+str(b)
  f_b.__name__='lf'+str(b)

  return f_b

def evaluate_scheme(scheme,X_train,X_test,y_train,y_test,seed,metrics,epochs,
                    batch_size,l1_size,l2_size,verbose):
  #try:
  resampler=(lambda a,b:(a.copy(),b.copy())) if scheme[1] is None else scheme[1]

  print(resampler)
  print(X_train.shape,y_train.shape)
  X_sel,y_sel=resampler(X_train,y_train)
  print(X_sel.shape,y_sel.shape)
  m=mk_two_layer_perceptron(X_sel,scheme[0],seed,metrics=metrics,
                            l1_size=l1_size,l2_size=l2_size)

  t0=thread_time()
  if len(X_sel)!=len(y_sel):
    raise Exception('len(X_sel)=='+str(len(X_sel))+' but len(y_sel)=='+str(len(y_sel))+'!')
  m.fit(X_sel,y_sel,epochs=epochs,batch_size=batch_size,verbose=verbose)
  te=thread_time()-t0
  r=list(m.evaluate(X_test,y_test,batch_size=X_test.shape[0]))
  loss_name=str(scheme[0])
  resampler_name=str(scheme[1])
  ret=[loss_name,resampler_name]+r

  if verbose:
    print('Task done after',epochs,' epochs!','Results:',ret,file=stderr)

  return ret+[te]

def evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,
                     p=None,epochs=200,batch_size=32,verbose=0,l1_size=64,l2_size=32,
                     metrics=['accuracy','binary_accuracy']):
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
                [str(t) for t in metrics]+['thread time']

  def bench_scheme(s):
    return evaluate_scheme(s,X_train,X_test,y_train,y_test,seed,metrics,
                           epochs,batch_size,l1_size,l2_size,verbose)

  results=p.map(bench_scheme,schemes)
  ret=DataFrame(results,columns=res_col_names)
  try:
    ret['cts_precision']=ret[str(cts_tp)]/(ret[str(cts_tp)]+ret[str(cts_fp)])
    ret['cts_recall']=ret[str(cts_tp)]/(ret[str(cts_tp)]+ret[str(cts_fn)])
    ret['bin_precision']=ret[str(bin_tp)]/(ret[str(bin_tp)]+ret[str(bin_fp)])
    ret['bin_recall']=ret[str(bin_tp)]/(ret[str(bin_tp)]+ret[str(bin_fn)])
    return ret
  except Exception as e:
    ret['error']=True
    return ret

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

# the is the f-beta Im using in this study
#     FbetaSurrogatePenalty
class FbetaSurrogatePenalty(Loss):
    def __init__(self, beta, thresh=0.5, fp_penalty=1.0, fn_rate=0.1):
        super(FbetaSurrogatePenalty, self).__init__()
        self.beta = beta
        self.thresh = thresh
        self.fp_penalty = fp_penalty
        self.fn_rate = fn_rate
        self.__name__='FbetaSurrogatePenalty'

    def call(self, y_true, y_pred):
        # Calculate additional metrics and terms
        P = reduce_sum(cast(y_true, float32))  # Ensure y_true is cast to float32
        N = reduce_sum(cast(1 - y_true, float32))  # Ensure 1 - y_true is cast to float32
        theta = (1 - P) * 1.0 / P

        if y_pred is not None:
            pred = cast(y_pred > self.thresh, dtype=float32)
            TP = reduce_sum(pred * cast(y_true, float32) * y_pred) / P
            TN = reduce_sum((1 - pred) * (1 - cast(y_true, float32)) * (1 - y_pred)) / N
            FP = reduce_sum(pred * (1 - cast(y_true, float32)) * y_pred) / P

            # Apply penalty for false positives
            penalty = self.fp_penalty * FP

            # Calculate the desired fixed rate of false negatives
            fixed_FN = self.fn_rate * P

            # Calculate F-measure with penalty for false positives
            f_measure = (1 + self.beta**2) * TP / (self.beta**2 + theta + TP - theta * TN + penalty)
            #f_measure = (1 + self.beta**2) * TP / (self.beta**2 + theta + TP - theta * TN )

            # Add a penalty term to maintain the fixed rate of false negatives
            fn_penalty = maximum(0.0, P - fixed_FN)
            f_measure += fn_penalty

        return -f_measure

class FbetaTPFP(Loss):
    def __init__(self, beta, thresh=0.5):
        super(FbetaTPFP, self).__init__()
        self.beta = beta
        self.thresh = thresh
        self.__name__='CombinedLoss'

    def call(self, y_true, y_pred):
        # Calculate additional metrics and terms
        P = reduce_sum(cast(y_true, float32))  # Ensure y_true is cast to float32
        N = reduce_sum(cast(1 - y_true, float32))  # Ensure 1 - y_true is cast to float32
        theta = (1 - P) * 1.0 / P

        if y_pred is not None:
            pred = cast(y_pred > self.thresh, dtype=float32)
            TP = reduce_sum(pred * cast(y_true, float32) * y_pred) / P
            TN = reduce_sum((1 - pred) * (1 - cast(y_true, float32)) * (1 - y_pred)) / N

            f_measure = (1 + self.beta**2) * TP / (self.beta**2 + theta + TP - theta * TN)

        return -f_measure

# MCC correlation coefficient Surrogate Loss
class MCC_Loss(Loss):
    def __init__(self):
        super(MCC_Loss, self).__init__()

    def call(self, y_true, y_pred):
        targets = cast(y_true, dtype=float32)
        inputs = cast(y_pred, dtype=float32)
        tp = reduce_sum(multiply(inputs, targets))
        tn = reduce_sum(multiply((1 - inputs), (1 - targets)))
        fp = reduce_sum(multiply(inputs, (1 - targets)))
        fn = reduce_sum(multiply((1 - inputs), targets))
        epsilon = 1e-7

        numerator = tp * tn - fp * fn
        denominator = sqrt((tp + fp + epsilon) * (tp + fn + epsilon) * (tn + fp + epsilon) * (tn + fn + epsilon))
        fpr = fp / (fp + tn + epsilon)  # False Positive Rate calculation

        penalty_term = tfabs(fpr - 0.2)
        mcc = divide(numerator, denominator)
        #mcc = mcc + 1.0 * penalty_term

        return -mcc


# Combined MCC and Fbeta Loss
class CombinedLoss(Loss):
    def __init__(self,fbeta_weight, mcc_weight=1.0,):
        super(CombinedLoss, self).__init__()
        self.mcc_weight = mcc_weight
        self.fbeta_weight = fbeta_weight
        self.mcc_loss = MCC_Loss()
        self.fbeta_loss = FbetaTPFP(beta=2, thresh=0.6)
        self.__name__='CombinedLoss'

    def call(self, y_true, y_pred):
        mcc_term = self.mcc_weight * self.mcc_loss(y_true, y_pred)
        fbeta_term = self.fbeta_weight * self.fbeta_loss(y_true, y_pred)
        combined_loss = mcc_term + fbeta_term
        #tf.print("Fbeta term=", fbeta_term)
        #tf.print("Mcc term=", mcc_term)
        #tf.print(" combined_loss=", combined_loss)
        #print("... ")
        #return combined_loss
        return fbeta_term
