import keras
import tensorflow as tf
import numpy as np

map_decs = {"incr":0, "decr":1, "noact":2, "noact_incr":2, "noact_decr":3}

class MeanAbsoluteDirectionalLoss(keras.losses.Loss):
    def call(self, y_true,y_pred):
        return keras.ops.mean(keras.ops.multiply(-1,keras.ops.multiply(keras.ops.sign(keras.ops.multiply(y_true,y_pred)),keras.ops.abs(y_true))))

class CustomCategoricalEntropyLoss(keras.losses.Loss):
    def call(self, y_true,y_pred):
        y_true = tf.reshape(tf.argmax(y_true,axis=1), shape=(-1, 1))
        y_is_three = tf.cast(tf.greater(y_true,2),tf.int64)
        y_true = keras.utils.to_categorical(tf.subtract(y_true,y_is_three),3)
        return keras.losses.categorical_crossentropy(y_true,y_pred)
        return keras.ops.mean(keras.ops.square(y_true-y_pred))

class CustomAccuracy(keras.metrics.Metric):
    def __init__(self, name="betting_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="pp", initializer="zeros")
        self.total = self.add_weight(name="pp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_correct = tf.greater(tf.sign(tf.multiply(y_pred,y_true)),0)
        is_incorrect = tf.less_equal(tf.sign(tf.multiply(y_pred,y_true)),0)
        num_correct = tf.reduce_sum(tf.cast(is_correct,tf.float32))
        num_incorrect = tf.reduce_sum(tf.cast(is_incorrect,tf.float32))
        self.correct.assign_add(num_correct)
        self.total.assign_add(num_correct)
        self.total.assign_add(num_incorrect)

    def result(self):
        return tf.divide(self.correct,self.total)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.correct.assign(0.0)
        self.total.assign(0.0)

class BetUp(keras.metrics.Metric):
    def __init__(self, name="betting_up", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos = self.add_weight(name="pp", initializer="zeros")
        self.total = self.add_weight(name="pp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_correct = tf.greater(y_pred,0)
        is_incorrect = tf.less_equal(y_pred,0)
        num_pos = tf.reduce_sum(tf.cast(is_correct,tf.float32))
        num_neg = tf.reduce_sum(tf.cast(is_incorrect,tf.float32))
        self.pos.assign_add(num_pos)
        self.total.assign_add(num_pos)
        self.total.assign_add(num_neg)

    def result(self):
        return tf.divide(self.pos,self.total)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.pos.assign(0.0)
        self.total.assign(0.0)



class ProfitPerformance(keras.metrics.Metric):
    def __init__(self, name="profit_performance", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_incr = self.add_weight(name="pp", initializer="zeros")
        self.true_decr = self.add_weight(name="pp", initializer="zeros")
        self.false_incr = self.add_weight(name="pp", initializer="zeros")
        self.false_decr = self.add_weight(name="pp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred,axis=1), shape=(-1, 1))
        y_true = tf.reshape(tf.argmax(y_true,axis=1), shape=(-1, 1))

        y_true_incr = tf.equal(y_true,map_decs["incr"])
        y_true_decr = tf.equal(y_true,map_decs["decr"])
        y_true_noact_incr = tf.equal(y_true,map_decs["noact_incr"])
        y_true_noact_decr = tf.equal(y_true,map_decs["noact_decr"])

        y_pred_incr = tf.equal(y_pred,map_decs["incr"])
        y_pred_decr = tf.equal(y_pred,map_decs["decr"])
        y_pred_noact = tf.equal(y_pred,map_decs["noact"])

        true_incr = tf.logical_or(tf.logical_and(y_pred_incr,y_true_incr),
                                  tf.logical_and(y_pred_incr,y_true_noact_incr))
        true_decr = tf.logical_or(tf.logical_and(y_pred_decr,y_true_decr),
                                  tf.logical_and(y_pred_decr,y_true_noact_decr))
        false_incr = tf.logical_or(tf.logical_and(y_pred_incr,y_true_decr),
                                   tf.logical_and(y_pred_incr,y_true_noact_decr))
        false_decr = tf.logical_or(tf.logical_and(y_pred_decr,y_true_incr),
                                   tf.logical_and(y_pred_decr,y_true_noact_incr))

        self.true_incr.assign_add(tf.reduce_sum(tf.cast(true_incr, tf.float32)))
        self.true_decr.assign_add(tf.reduce_sum(tf.cast(true_decr, tf.float32)))
        self.false_incr.assign_add(tf.reduce_sum(tf.cast(false_incr, tf.float32)))
        self.false_decr.assign_add(tf.reduce_sum(tf.cast(false_decr, tf.float32)))

    def result(self):
        return tf.divide(tf.add(self.true_incr,self.true_decr),
                         tf.add(tf.add(self.true_incr,self.true_decr),
                                tf.add(self.false_incr,self.false_decr)))

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_incr.assign(0.0)
        self.true_decr.assign(0.0)
        self.false_incr.assign(0.0)
        self.false_decr.assign(0.0)

def transaction_ratio(y_pred):
    incr,decr,noact = map_decs["incr"],map_decs["decr"],map_decs["noact"]
    y_pred = np.array(y_pred)

    trans = np.logical_or(y_pred==incr,y_pred==decr)
    num_trans = np.array(list(map(int,trans))).sum()

    not_trans = (y_pred==map_decs["noact"])
    num_not_trans = np.array(list(map(int,not_trans))).sum()

    return num_trans,(num_trans+num_not_trans)

def profit_performance(y_true,y_pred):
    incr,decr,noact = map_decs["incr"],map_decs["decr"],map_decs["noact"]
    noact_incr,noact_decr = map_decs["noact_incr"],map_decs["noact_decr"]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_incr = (y_true==incr)
    y_true_decr = (y_true==decr)
    y_true_noact_incr = (y_true==noact_incr)
    y_true_noact_decr = (y_true==noact_decr)

    y_pred_incr = (y_pred==incr)
    y_pred_decr = (y_pred==decr)
    y_pred_noact = (y_pred==noact)

    true_incr  = np.logical_or(np.logical_and(y_pred_incr,y_true_incr),
                               np.logical_and(y_pred_incr,y_true_noact_incr))
    true_decr  = np.logical_or(np.logical_and(y_pred_decr,y_true_decr),
                               np.logical_and(y_pred_decr,y_true_noact_decr))
    false_incr = np.logical_or(np.logical_and(y_pred_incr,y_true_decr),
                               np.logical_and(y_pred_incr,y_true_noact_decr))
    false_decr = np.logical_or(np.logical_and(y_pred_decr,y_true_incr),
                               np.logical_and(y_pred_decr,y_true_noact_incr))

    TrueIncr  = np.array(list(map(int,true_incr))).sum()
    TrueDecr  = np.array(list(map(int,true_decr))).sum()
    FalseIncr = np.array(list(map(int,false_incr))).sum()
    FalseDecr = np.array(list(map(int,false_decr))).sum()

    #if TrueIncr+TrueDecr+FalseIncr+FalseDecr != len(y_true):
        #print("Warning: something went wrong in counting.")
        #print(f"true_incr: {TrueIncr}\ntrue_decr:{TrueDecr}")
        #print(f"false_incr: {FalseIncr}\nfalse_decr:{FalseDecr}")

    return 1.0*(TrueIncr+TrueDecr)/(TrueIncr+TrueDecr+FalseIncr+FalseDecr)



