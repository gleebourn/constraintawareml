from sys import argv,stdout
from argparse import ArgumentParser
from pandas import read_csv

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks,\
                                    EditedNearestNeighbours
from imblearn.combine import SMOTETomek

from consawml import FbetaMetric,mk_two_layer_perceptron,MatthewsCorrelationCoefficient,\
                     preproc_bin_class,mk_F_beta, evaluate_schemes,undersample_positive,\
                     MCCWithPenaltyAndFixedFN_v2,MCCWithPenaltyAndFixedFN_v3, precision_metric,\
                     recall_metric,binary_precision_metric,binary_recall_metric

p=ArgumentParser(description='''Benchmarks training algorithms on a given dataset and generates reports.

  May most of this into another function - but one would still want to be able to
  pass it custom loss and resampling schemes, for example.
  ''')
p.add_argument('-s','--seed',type=int,default=42,help='Random seed for reproducibility')
p.add_argument('-i',help='Input data filename, if data is to be train-test split automatically')
p.add_argument('-o',type=str,default=None,help='Output file for results, defaults to stdout')
p.add_argument('-e','--epochs',type=int,default=6,help='Number of epochs for the fitting algorithm')
p.add_argument('-train',help='Input training data filename')
p.add_argument('-test',help='Input testing data filename')
args=p.parse_args()
seed=args.seed
out_file=args.o
input_filename=args.i
train_filename=args.train
test_filename=args.test
epochs=args.epochs
print()
print()

#Split into training and testing

if input_filename:
  X_train,X_test,y_train,y_test = preproc_bin_class(input_filename,seed)
elif train_filename and test_filename:
  Xy_train,Xy_test=read_csv(train_filename),read_csv(test_filename)
  P_X=lambda A:A.drop(labels=['id','attack_cat','label'],axis=1).select_dtypes('number')
  P_y=lambda A:A['label']
  X_train,X_test,y_train,y_test=P_X(Xy_train),P_X(Xy_test),P_y(Xy_train),P_y(Xy_test)
else:
  p.print_help()
  exit(1)
  

print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

print()
print('Generate some custom metrics for various falues of beta')
fh=mk_F_beta(.5)
f1=mk_F_beta(1)
f2=mk_F_beta(2)
f3=mk_F_beta(3)
f4=mk_F_beta(4)
#metrics=['accuracy','binary_accuracy']
metrics=['accuracy','binary_accuracy',precision_metric,recall_metric,
         binary_precision_metric,binary_recall_metric,fh,f1,f2,f3,f4]

losses_to_evaluate=[f1,f2,f3,'binary_crossentropy','mean_squared_logarithmic_error',
                    'kl_divergence','mean_squared_error',
                    MCCWithPenaltyAndFixedFN_v2(),MCCWithPenaltyAndFixedFN_v3()]

resampling_algorithms_to_evaluate=[None,SMOTETomek().fit_resample,RandomOverSampler().fit_resample,
                                   SMOTE().fit_resample,ADASYN().fit_resample,RandomUnderSampler().fit_resample,
                                   NearMiss().fit_resample,TomekLinks().fit_resample,
                                   EditedNearestNeighbours().fit_resample]

#Test all combos of loss with resamplers
schemes=[(a,b) for a in losses_to_evaluate for b in resampling_algorithms_to_evaluate]
'''
schemes=[(a,b) for a in [f1,f2] for b in [None,SMOTETomek().fit_resample]]
'''
a=evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,epochs=epochs,
                   metrics=metrics)

a.to_csv(out_file)
  


#with open(argv[2],'w',newline='') as f:
#  w=writer(f)
#  w.writerow(['loss function','resampling scheme',a[0][1][0].name]+
#             [str(t.name) for t in a[0][1][1].metrics])
#  for s,z in zip(schemes,a):
#    w.writerow(list(s)+[str(t) for t in z[0]])
