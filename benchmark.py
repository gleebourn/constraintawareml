from sys import argv
from csv import writer
if len(argv)!=3:
  print('Usage:',argv[0],'[DATASET FILE LOCATION] [OUTPUT CSV LOCATION]')
  exit(1)
seed=42

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks,\
                                    EditedNearestNeighbours
from imblearn.combine import SMOTETomek

from consawml import FbetaMetric,mk_two_layer_perceptron,MatthewsCorrelationCoefficient,\
                     preproc_bin_class,mk_F_beta, evaluate_schemes,undersample_positive,\
                     MCCWithPenaltyAndFixedFN_v2,MCCWithPenaltyAndFixedFN_v3, precision_metric,\
                     recall_metric,binary_precision_metric,binary_recall_metric

print()
print()

#Split into training and testing

X_train,X_test,y_train,y_test = preproc_bin_class(argv[1],seed)
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

print()
print()
fh=mk_F_beta(.5)
f1=mk_F_beta(1)
f2=mk_F_beta(2)
f3=mk_F_beta(3)
f4=mk_F_beta(4)
#metrics=['accuracy','binary_accuracy']
metrics=['accuracy','binary_accuracy',precision_metric,recall_metric,
         binary_precision_metric,binary_recall_metric,fh,f1,f2,f3,f4]

#Test all combos of loss with resamplers
schemes=[(a,b) for a in [f1,f2,f3,'binary_crossentropy','mean_squared_logarithmic_error',
                         'kl_divergence','mean_squared_error','cosine_similarity',
                         MCCWithPenaltyAndFixedFN_v2(),MCCWithPenaltyAndFixedFN_v3()]\
               for b in [None,SMOTETomek().fit_resample,RandomOverSampler().fit_resample,
                         SMOTE().fit_resample,ADASYN().fit_resample,RandomUnderSampler().fit_resample,
                         NearMiss().fit_resample,TomekLinks().fit_resample,
                         EditedNearestNeighbours().fit_resample]]
'''
schemes=[(a,b) for a in [f1,f2] for b in [None,SMOTETomek().fit_resample]]
'''
a=evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,epochs=30,
                   metrics=metrics)

print()
with open(argv[2],'w',newline='') as f:
  w=writer(f)
  w.writerow(['loss function','resampling scheme',a[0][1][0].name]+
             [str(t.name) for t in a[0][1][1].metrics])
  for s,z in zip(schemes,a):
    w.writerow(list(s)+[str(t) for t in z[0]])
