from sys import argv
if len(argv)!=2:
  print('Usage:',argv[0],'[DATASET FILE LOCATION]')
  exit(1)
seed=42

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

print('Undersample to set proportion positive=0.1')
U,u=undersample_positive(X_train,y_train,seed,p=.1)
print('Undersampled data shape:',U.shape,u.shape)
print('Undersampled data number of positives:',u[u].sum())

print()
print()

schemes=[(f_beta_1,None),
         (mk_F_beta(2),None),
         (mk_F_beta(3),None),
         ('binary_crossentropy',None),
         ('mean_squared_logarithmic_error',None),
         ('kl_divergence',None),
         ('mean_squared_error',None),
         ('cosine_similarity',None)]
a=evaluate_schemes(schemes,X_train,X_test,y_train,y_test,seed,epochs=30)

print()
print()
print('loss function,resampling scheme,loss value,accuracy,binary accuracy,precision, F_\\beta')
for s,z in zip(schemes,a):
  print(s[0],',',s[1],',','',','.join([str(t) for t in z]))
