from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter

from cal.consawml import evaluate_schemes,mk_two_layer_perceptron,preproc_bin_class,\
                         mk_F_beta,mk_rebalanced_lq,mk_rebalanced_ls,\
                         mk_log_F_beta,mk_gp_beta,mk_gn_beta,\
                         FbetaSurrogatePenalty,CombinedLoss,\
                         cts_fp,cts_fn,cts_tp,cts_tn,bin_fp,bin_fn,bin_tp,bin_tn

from cal.mmet import FbetaMetric,MCCWithPenaltyAndFixedFN_v2,MCCWithPenaltyAndFixedFN_v3

from cal.rsynth import near_random_path

from numpy import linspace


available_losses={'binary_crossentropy':'binary_crossentropy',
                  'mean_squared_logarithmic_error':'mean_squared_logarithmic_error',
                  'kl_divergence':'kl_divergence','mean_squared_error':'mean_squared_error',
                  'mcc_fixed_p_fn2':MCCWithPenaltyAndFixedFN_v2(),
                  'mcc_fixed_p_fn3':MCCWithPenaltyAndFixedFN_v3()}

available_param_losses={'mk_F_beta':mk_F_beta,'mk_rebalanced_ls':mk_rebalanced_ls,
                        'mk_gp_beta':mk_gp_beta,'mk_gn_beta':mk_gn_beta,
                        'mk_rebalanced_lq':mk_rebalanced_lq,'mk_log_F_beta':mk_log_F_beta,
                        'FbetaSurrogatePenalty':FbetaSurrogatePenalty,'Combinedloss':CombinedLoss}

beta_vals=[.5,1.,2.,3.,4.]

for mk_F in available_param_losses.values():
  for b in beta_vals:
    l=mk_F(b)
    available_losses[l.__name__]=l

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks,\
                                    EditedNearestNeighbours
from imblearn.combine import SMOTETomek

def error_out(s):
  s.p.print_help()
  exit(1)

available_resampling_algorithms={'None':None,'SMOTETomek':SMOTETomek().fit_resample,
                                 'random_oversampler':RandomOverSampler().fit_resample,
                                 'SMOTE':SMOTE().fit_resample,'ADASYN':ADASYN().fit_resample,
                                 'random_undersampler':RandomUnderSampler().fit_resample,
                                 'near_miss':NearMiss().fit_resample,'Tomek_links':TomekLinks().fit_resample,
                                 'edited_nearest_neighbours':EditedNearestNeighbours().fit_resample}

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

from pandas import read_csv,DataFrame

#default_mets=['accuracy','binary_accuracy',cts_precision,cts_recall,binary_precision,binary_recall,
#              cts_fp,cts_fn,cts_tp,cts_tn,bin_fp,bin_fn,bin_tp,bin_tn,]

# keras metrics need more niceity of thought than I possess - calculate them based on the confusion matrices.
default_mets=['accuracy','binary_accuracy',cts_fp,cts_fn,cts_tp,cts_tn,bin_fp,bin_fn,bin_tp,bin_tn]

class Benchmarker():
    
  def __init__(self,passed_params=False,metrics=default_mets):
    self.p=ArgumentParser(description='Benchmarks training algorithms on a given dataset.',
                        formatter_class=ArgumentDefaultsHelpFormatter)
    self.p.add_argument('-s','--seed',type=int,default=42,help='Random seed for reproducibility')
    self.p.add_argument('-i',help='Input data filename, if data is to be train-test split automatically')
    self.p.add_argument('-o',type=str,default=None,help='Output file prepender for saved file, defaults to stdout')
    self.p.add_argument('-e','--epochs',type=int,default=30,help='Number of epochs for the fitting algorithm')
    self.p.add_argument('-train',help='Input training data filename')
    self.p.add_argument('-test',help='Input testing data filename')
    self.p.add_argument('-rows',type=int,help='Number of random rows',default=1000)
    self.p.add_argument('-l',nargs='+',default=[],help='Loss functions to train on')
    self.p.add_argument('-r',nargs='+',default=[],help='Resampling schemes to train on')
    self.p.add_argument('-a','--print-algs',action='store_true',help='Print available algorithms')
    self.p.add_argument('-n1',type=int,default=128,help='Layer 1 size')
    self.p.add_argument('-n2',type=int,default=128,help='Layer 2 size')
    self.p.add_argument('-v',type=int,default=0,help='Verbosity')
    self.p.add_argument('-b',type=int,default=32,help='Batch size')
    self.p.add_argument('-u',type=float,default=None,help='Undersample the training and test data to balance u')
    self.p.add_argument('-p',type=float,default=False,help='Generate a synthetic dataset with proportion p')
    self.p.add_argument('-paraloss',type=str,default=None,help='Test a parametrised loss function over a range of values')
    if passed_params:
      self.a=self.p.parse_args([])
      for k,v in passed_params.items():
        setattr(self.a,k,v)

    else:
      self.a=self.p.parse_args()
      
    if self.a.print_algs:
      print()
      print('====================================')
      print('=======Resampling algorithms:=======')
      for k,v in available_resampling_algorithms.items():
        print(k,':',v)
      print('=============Losses:================')
      for k,v in available_losses.items():
        print(k,':',v)
      print('====================================')
      exit(0)
    
    self.metrics=metrics
  
  def get_xy(self):
    
    if self.a.i: #We split the data into training and testing ourselves
      self.X_train,self.X_test,self.y_train,self.y_test = preproc_bin_class(self.a.i,self.a.seed)
    
    elif self.a.train and self.a.test: #Data already split into test and train
      Xy_train,Xy_test=read_csv(self.a.train),read_csv(self.a.test)
      P_X=lambda A:A.drop(labels=['id','attack_cat','label'],axis=1).select_dtypes('number')
      P_y=lambda A:A['label'].astype(float)
      self.X_train,self.X_test,self.y_train,self.y_test=P_X(Xy_train),P_X(Xy_test),P_y(Xy_train),P_y(Xy_test)
    
    elif self.a.p:
      print('Generating synthetic data...')
      synth_dim=2
      synth_std=.00001
      path_len=400
      data_len=self.a.rows
      X,y=near_random_path(synth_dim,synth_std,path_len,data_len,
                           self.a.p,seed=self.a.seed,regularity=2)
      y=y.astype(float)
      self.X_train=DataFrame(X[:int(.7*data_len)])
      self.X_test=DataFrame(X[int(.7*data_len):])
      self.y_train=y[:int(.7*data_len)]
      self.y_test=y[int(.7*data_len):]
      print('Done generating!')
    
    else:
      error_out(self)
    
    print('X_train.shape:',self.X_train.shape)
    print('X_test.shape:',self.X_test.shape)
    print('y_train.shape:',self.y_train.shape)
    print('y_test.shape:',self.y_test.shape)

  def synthetic_undersample(self):
    print('Undersampling to make positive proportion=',args.u)
    self.X_train,self.y_train=undersample_positive(self.X_train,self.y_train,self.a.seed,self.a.u)
    self.X_test,self.y_test=undersample_positive(self.X_test,self.y_test,self.a.seed,self.a.u)
    print('New stats:')
    self.print_data_overview()
  
  def y_train_positive_proportion(self):
    return self.y_train.sum()/self.y_train.shape[0]
  
  def y_test_positive_proportion(self):
    return self.y_test.sum()/self.y_test.shape[0]
  
  def print_data_overview(self):
    print('Training data shape:',self.X_train.shape)
    print('Testing data shape:',self.X_test.shape)
    print('Training data positive proportion:',self.y_train_positive_proportion())
    print('Testing data positive proportion:',self.y_test_positive_proportion())
  
  def generate_benchmarking_tasks(self,losses=None,resampling=None):
    if self.a.paraloss:
      mk_loss=available_param_losses[self.a.paraloss]
      self.mk_parametrised_loss(mk_loss,resampling=resampling)
    else:
      self.populate_benchmarking_tasks(losses=losses,resampling=resampling)

  def mk_parametrised_loss(self,mk_loss,loss_params=linspace(.5,2,num=20),resampling=[None]):
    losses=[mk_loss(i) for i in loss_params]
    self.populate_benchmarking_tasks(losses,resampling=resampling)

  def populate_benchmarking_tasks(self,losses,resampling):
    losses=losses or [available_losses[i]for i in self.a.l]
    resampling=resampling or [available_resampling_algorithms[j] for j in self.a.r]
    #Test all combos of loss with resamplers
    self.tasks=[(a,b) for a in losses for b in resampling]
  
  def benchmark(self):
    self.res=evaluate_schemes(self.tasks,self.X_train,self.X_test,self.y_train,self.y_test,
                              self.a.seed,epochs=self.a.epochs,batch_size=self.a.b,metrics=self.metrics,
                              l1_size=self.a.n1,l2_size=self.a.n2,verbose=self.a.v)

  def handle_results(self):
    out_file=self.a.o+'_e_'+str(self.a.epochs)+'_l1_'+str(self.a.n1)+'_l2_'+str(self.a.n2)
    if self.a.u:
      out_file+='_u_'+str(self.a.u)
    
    self.res.to_csv(out_file+'.csv')
    
    if self.a.p:
      self.X_train['y']=self.y_train
      self.X_test['y']=self.y_test
      self.X_train.to_csv(out_file+'_train.csv')
      self.X_test.to_csv(out_file+'_test.csv')
