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

f_beta_1=mk_F_beta(1)
f_beta_2=mk_F_beta(2)
f_beta_3=mk_F_beta(3)

#Split into training and testing

X_train,X_test,y_train,y_test = preproc_bin_class(argv[1],seed)
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

print()
print()
smote_tomek=SMOTETomek().fit_resample

#Test all combos of loss with resamplers
schemes=[(a,b) for a in [f_beta_1,mk_F_beta(2),mk_F_beta(3),'binary_crossentropy',
                         'mean_squared_logarithmic_error','kl_divergence',
                         'mean_squared_error','cosine_similarity']\
               for b in [None,SMOTETomek().fit_resample,RandomOverSampler().fit_resample,
                         SMOTE().fit_resample,ADASYN().fit_resample,RandomUnderSampler().fit_resample,
                         NearMiss().fit_resample,TomekLinks().fit_resample,
                         EditedNearestNeighbours().fit_resample]]
a=evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,epochs=30)

print()
print()
print('loss function,resampling scheme,loss value,accuracy,binary accuracy,precision, F_\\beta')
for s,z in zip(schemes,a):
  print(s[0],',',s[1],',','',','.join([str(t) for t in z]))
