from tensorflow import reduce_sum,cast,float32,maximum,cast,\
                       logical_and,logical_not,sqrt,multiply,divide
from tensorflow import abs as tfabs
from keras.backend import epsilon
from keras.metrics import Metric
from keras.losses import Loss

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
    #scaled_penalty_term = penalty_term / max_abs_penalty
    #scaled_fn_penalty = fn_penalty / max_abs_penalty

    #return scaled_penalty_term, scaled_fn_penalty

    alpha = 0.6

    # Adjust the final loss with the MCC and penalty terms
    final_loss = - alpha*mcc + (1-alpha)*(penalty_term + fn_penalty)                     # Original formuula

    return final_loss