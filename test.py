from sys import argv
if len(argv)!=2:
  print('Usage:',argv[0],'[DATASET FILE LOCATION]')
  exit(1)
seed=42

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks,\
                                    EditedNearestNeighbours
from imblearn.combine import SMOTETomek

from consawml import FbetaMetric,mk_two_layer_perceptron,MatthewsCorrelationCoefficient,\
                     preproc_bin_class,mk_F_beta, evaluate_schemes,undersample_positive

print()
print()

#Split into training and testing

X_train,X_test,y_train,y_test = preproc_bin_class(argv[1],seed)
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

print()
print()
f_b_half=mk_F_beta(.5)
f_b_1=mk_F_beta(1)
f_b_2=mk_F_beta(2)
f_b_3=mk_F_beta(3)
f_b_4=mk_F_beta(4)
metrics=['accuracy','binary_accuracy',f_b_half,f_b_1,f_b_2,f_b_3,f_b_4]

#Test all combos of loss with resamplers
schemes=[(a,b) for a in [f_b_1,f_b_2,f_b_3,'binary_crossentropy',
                         'mean_squared_logarithmic_error','kl_divergence',
                         'mean_squared_error','cosine_similarity']\
               for b in [None,SMOTETomek().fit_resample,RandomOverSampler().fit_resample,
                         SMOTE().fit_resample,ADASYN().fit_resample,RandomUnderSampler().fit_resample,
                         NearMiss().fit_resample,TomekLinks().fit_resample,
                         EditedNearestNeighbours().fit_resample]]
a=evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,epochs=30,
                   metrics=metrics)

print()
print()
print('loss function,resampling scheme,loss value,'+(','.join([str(m) for m in metrics])))
for s,z in zip(schemes,a):
  print(s[0],',',s[1],',','',','.join([str(t) for t in z]))
