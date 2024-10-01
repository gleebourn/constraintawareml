from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from pandas import read_csv,DataFrame

from sys import path
from pathlib import Path
path.append(str(Path('.').absolute()))

from cal.rsynth import near_random_path

from cal.consawml import FbetaMetric,mk_two_layer_perceptron,MatthewsCorrelationCoefficient,\
                        preproc_bin_class,mk_F_beta, evaluate_schemes,undersample_positive,\
                        MCCWithPenaltyAndFixedFN_v2,MCCWithPenaltyAndFixedFN_v3, precision_metric,\
                        recall_metric,binary_precision_metric,binary_recall_metric,available_losses,\
                        available_resampling_algorithms,precision_metric,recall_metric,\
                        binary_precision_metric,binary_recall_metric,fh,f1,f2,f3,f4

p=ArgumentParser(description='Benchmarks training algorithms on a given dataset.',
                 formatter_class=ArgumentDefaultsHelpFormatter)
p.add_argument('-s','--seed',type=int,default=42,help='Random seed for reproducibility')
p.add_argument('-i',help='Input data filename, if data is to be train-test split automatically')
p.add_argument('-o',type=str,default=None,help='Output file prepender for saved file, defaults to stdout')
p.add_argument('-e','--epochs',type=int,default=30,help='Number of epochs for the fitting algorithm')
p.add_argument('-train',help='Input training data filename')
p.add_argument('-test',help='Input testing data filename')
p.add_argument('-l',nargs='+',default=[],help='Loss functions to train on')
p.add_argument('-r',nargs='+',default=[],help='Resampling schemes to train on')
p.add_argument('-a','--print-algs',action='store_true',help='Print available algorithms')
p.add_argument('-n1',type=int,default=128,help='Layer 1 size')
p.add_argument('-n2',type=int,default=128,help='Layer 2 size')
p.add_argument('-v',type=int,default=0,help='Verbosity')
p.add_argument('-b',type=int,default=32,help='Batch size')
p.add_argument('-u',type=float,default=None,help='Undersample the training and test data to balance u')
p.add_argument('-p',type=float,default=False,help='Generate a synthetic dataset with proportion p')
args=p.parse_args()

if args.print_algs:
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

seed=args.seed
out_file=args.o
input_filename=args.i
train_filename=args.train
test_filename=args.test
epochs=args.epochs
losses_to_evaluate=args.l
resampling_algorithms_to_evaluate=args.r
l1_size=args.n1
l2_size=args.n2
verbosity=args.v
batch_size=args.b
synthetic_undersample=args.u
generate_synthetic=args.p

print()


if input_filename: #We split the data into training and testing ourselves
  X_train,X_test,y_train,y_test = preproc_bin_class(input_filename,seed)
elif train_filename and test_filename: #Data already split into test and train
  Xy_train,Xy_test=read_csv(train_filename),read_csv(test_filename)
  P_X=lambda A:A.drop(labels=['id','attack_cat','label'],axis=1).select_dtypes('number')
  P_y=lambda A:A['label']
  X_train,X_test,y_train,y_test=P_X(Xy_train),P_X(Xy_test),P_y(Xy_train),P_y(Xy_test)
elif generate_synthetic:
  print('Generating synthetic data...')
  synth_dim=64
  synth_std=.1
  path_len=200
  data_len=100000
  X,y=near_random_path(synth_dim,synth_std,path_len,data_len,
                       generate_synthetic,seed=seed,regularity=2)
  X_train=DataFrame(X[:int(.7*data_len)])
  X_test=DataFrame(X[int(.7*data_len):])
  y_train=y[:int(.7*data_len)]
  y_test=y[int(.7*data_len):]
  print('Done generating!')
else:
  p.print_help()
  exit(1)
  
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)
print('Training data positive proportion:',y_train.sum()/y_train.shape[0])
print('Testing data positive proportion:',y_test.sum()/y_test.shape[0])

if synthetic_undersample:
  print('Undersampling to make positive proportion=',synthetic_undersample)
  X_train,y_train=undersample_positive(X_train,y_train,seed,synthetic_undersample)
  X_test,y_test=undersample_positive(X_test,y_test,seed,synthetic_undersample)

print()

metrics=['accuracy','binary_accuracy',precision_metric,recall_metric,
         binary_precision_metric,binary_recall_metric,fh,f1,f2,f3,f4]

#Test all combos of loss with resamplers
schemes=[(available_losses[a],available_resampling_algorithms[b])\
         for a in losses_to_evaluate for b in resampling_algorithms_to_evaluate]

a=evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,epochs=epochs,batch_size=batch_size,
                   metrics=metrics,l1_size=l1_size,l2_size=l2_size,verbose=verbosity)

if isinstance(out_file,str):
  out_file+='_e_'+str(epochs)+'_l1_'+str(l1_size)+'_l2_'+str(l2_size)
  if synthetic_undersample:
    out_file+='_u_'+str(synthetic_undersample)
  out_file+='.csv'

a.to_csv(out_file)

if generate_synthetic:
  X_train['y']=y_train
  X_test['y']=y_test
  X_train.to_csv('train_'+out_file)
  X_test.to_csv('test_'+out_file)
