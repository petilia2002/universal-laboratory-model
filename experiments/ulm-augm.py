
import fdb
import numpy as np
import math
import sys 
import pickle
import datetime 
import os

import keras
from keras import Model
from keras.layers import Input, Dense, Embedding
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc

import tensorflow
import tensorflow.python.keras.backend as K

from configparser import ConfigParser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data import MASKED_VALUE
from layers import RealValueLayer, MaskLayerLeft, MaskLayerRight, MASKED_VALUE 
from callbacks import MessagerCallback
from utils import calc_confusion_matrix
from tools import get_variants, DataGenerator

import analyte
import inspect

config_object = ConfigParser()
config_object.read("../config.ini")

user_info = config_object["USERINFO"]
server_config = config_object["SERVERCONFIG"]

dsn = server_config["dsn"]
user = user_info["user"]
password = user_info["password"]

con = fdb.connect(dsn=dsn, user=user, password=password);

print("GPU: ", tensorflow.config.list_physical_devices('GPU'))

BATCH_SIZE=128

# List all classes of analyte.py and create objects.
# Then create variables for each object as it's viewname
# so it could be accessed from within the code 
# like uric without explicitly writing uric = Uric();

for name, cls_obj in inspect.getmembers(analyte, inspect.isclass):
    if cls_obj.__module__ == analyte.__name__ and cls_obj.__name__ != "Analyte":
       instance = cls_obj()
       name = instance.getViewName();    
       globals()[name] = instance;
   

#all analytes we can use during the training
source = [gender, age, hgb, rbc, mcv, plt, 
          wbc, neut, lymph, eo, baso, mono,
          mid, gra, b12, folic, ast, alt, 
          bil_direct, bil_indirect, bil_total, 
          crea, urea, pro, ldg, alb, crp, 
          chol, glu, ferritin, uric, hba1c, psa]

#tests we want to predict 
tests = [ chol, alt, ast, hba1c ];

X = [];
W = [];
Y = [];

sql = " is not null or ".join([ ("a." + analyte.getViewName()) for analyte in tests]) + " is not null";
sel = ", ".join([ ("a." + analyte.getViewName()) for analyte in source]);

y_map = {obj.getId() : i for i, obj in enumerate(tests)}
x_map = {obj.getId() : i for i, obj in enumerate(source)}

mid_id = x_map[mid.getId()];
gra_id = x_map[gra.getId()];

eo_id = x_map[eo.getId()];
mono_id = x_map[mono.getId()];
baso_id = x_map[baso.getId()];
lymph_id = x_map[lymph.getId()];
glu_id = x_map[glu.getId()];

for data in con.cursor().execute(f"""select first 1000 {sel}, lab_id from lab2 a 
                                     where ({sql}) and a.logdate <= '2025/04/01' """):
   data = list(data);

   lab_id = data[-1];
   data = data[:-1];

   #Lab's 9 glucose values are unreliable.
   if lab_id == 9: 
       data[glu_id] = None;   

   y = [MASKED_VALUE] * len(tests);       
   x = [ 0 ] * len(source);
   w = [ 0 ] * len(source);
 
   #add mid/gra if needed
   if data[mid_id] is None and \
      data[eo_id] is not None and data[baso_id] is not None and data[mono_id] is not None: 

       data[mid_id] = float(data[eo_id]) + float(data[mono_id]) + float(data[baso_id]);

       if data[lymph_id] is not None:
          data[gra_id] = 100.0 - float(data[lymph_id]) - data[mid_id];

   z = -1;
   sex = 0;
   y_cnt = 0;
 
   for ind, value in enumerate(data):

      analyte_id = source[ind].getId();
      if analyte_id == gender.getId():
         sex = value;

      v = source[ind].up(value);
      if v is not None:
          z = z + 1;

          x[z] = v;
          w[z] = analyte_id;
          
          if analyte_id in y_map: 
              y[y_map[analyte_id]] = source[ind].positive(value, sex);
              y_cnt = y_cnt + 1;

   if y_cnt:

      X.append(x);
      W.append(w);
      Y.append(y);      

for analyte in source:
   print(analyte.getName(), analyte.getMinimum(), analyte.getMaximum());

with open("e3.pickle", 'wb') as file:
   pickle.dump(source, file);

print("Total: ", len(X));

gpt = np.load("./embeddings.npy");
dims = gpt.shape;
emb_size = dims[1];

print(dims);

l_yandex = Input(name='input_analytes', shape=(None,));
l_embed = Embedding(name='embedding', input_dim = dims[0], output_dim = dims[1])(l_yandex);

embed = Model(l_yandex, l_embed, name="gpt");
embed.set_weights([gpt]);
embed.trainable = False;

embed.summary();

l_in = Input(name='values', shape=(len(source),))
l_words = Input(name='analytes', shape=(len(source),))
l_mask = Input(name='mask', shape=(len(source),))

l_pred = Input(name='outcomes', shape=(None,))

l_emb = embed(l_words)
l_tests = embed(l_pred);

embed.trainable = False;

l_m1 = MaskLayerLeft()(l_mask);
l_m2 = MaskLayerRight()([l_pred, l_mask]);

l_v = RealValueLayer(embedding_size = emb_size, name = "real_value_embedding")([l_emb, l_in]);
l_pos = Embedding(name='yandex_bias', input_dim = dims[0], output_dim = emb_size)(l_words);
l_v = l_v + l_pos;

l_p = Embedding(name='yandex_bias_pred', input_dim = dims[0], output_dim = emb_size)(l_pred);
l_tests = l_tests + l_p;

l_enc = keras.layers.MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.1)(query=l_v, value=l_v, key=l_v, attention_mask=l_m1);
l_dec = keras.layers.MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.1)(query=l_tests, value=l_enc, key=l_enc, attention_mask =l_m2);

l_o = keras.layers.TimeDistributed(Dense(units = 16, activation="relu"))(l_dec);

l_out_class = keras.layers.TimeDistributed( Dense(units= 1, activation = 'sigmoid'))(l_o);
l_out_regression = keras.layers.TimeDistributed( Dense(units= 1, activation = 'sigmoid'))(l_o);


def cbcLoss(y_true, y_pred):
   mask = K.cast_to_floatx(K.not_equal(y_true[0], MASKED_VALUE))

   y_class = tensorflow.clip_by_value(y_pred[0], K.epsilon(), 1.0 - K.epsilon() );
   r_class = y_true[0] * K.log(y_class) + (1.0 - y_true[0]) * K.log(1.0 - y_class);
   r_class = -tensorflow.reduce_mean(r_class * mask );

   r_mse = tensorflow.reduce_mean(tensorflow.math.square((y_true[1] - y_pred[1]) * mask))

   return r_class + 0.1 * r_mse;  

model = Model([l_in, l_words, l_mask, l_pred], [l_out_class, l_out_regression]);

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4) , loss= cbcLoss)
model.summary()


ids = np.arange(0, len(X), 1, dtype=int);
np.random.shuffle(ids);

train_size = (int)(len(ids) * 0.8)
test_size = len(ids) - train_size;

print("Training sizes: ", train_size);
print("Test sizes: ", test_size);

train_gen = DataGenerator(0, int(math.floor(0.95*train_size)), ids, X, W, Y, \
                    source, tests, x_map, y_map, \
                    MASKED_VALUE, BATCH_SIZE);

val_gen= DataGenerator(int(math.floor(0.95*train_size)), train_size, ids, X, W, Y, \
                        source, tests, x_map, y_map, \
                        MASKED_VALUE, BATCH_SIZE);
val_gen.noShuffle();

test_gen= DataGenerator(train_size, len(X), ids, X, W, Y, \
                        source, tests, x_map, y_map, \
                        MASKED_VALUE, BATCH_SIZE);

#model.load_weights("ulm-base.weights.h5");

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=100, verbose=1, mode="auto",
    baseline=None, restore_best_weights=True, start_from_epoch=0,
)

reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor = 0.9, patience = 20, verbose = 0, mode="auto",
    min_delta=0.00001, cooldown=0, min_lr= 1e-7)


print("Validation batches: ", len(val_gen));

model.fit(x = train_gen, epochs=100, shuffle=True, validation_data = val_gen, 
          callbacks = [tensorboard_callback, early_stopping, reduce, tensorflow.keras.callbacks.TerminateOnNaN(), MessagerCallback()]) 

model.save_weights("ulm-base.weights.h5", True);

train_gen.noShuffle();
test_gen.noShuffle();

y_pred = model.predict(train_gen)
y2_pred = model.predict(test_gen)

def calcRocs(analyte, key, label, ans):
    
    q = [];
    v = [];
    t_r = []
    t_p = [];

    qe = [];
    qr = [];
    q_r = [];
    q_p = [];

    for j in range(len(train_gen)):
         
        a, b = train_gen[j]; 

        m_pred_train = a[3];
        w_train = a[1];

        y_train = b[0];
        r_train = b[1];

        for k in range(len(m_pred_train)):    

           if analyte not in m_pred_train[k]: continue;
           prop = np.where(m_pred_train[k] == analyte)[0][0]
       
           if len(set(w_train[k]).intersection(key)) > 0: continue;

           if y_train[k, prop] != MASKED_VALUE:
              q.append( y_train[k, prop] );
              v.append( y_pred[0][j*BATCH_SIZE + k, prop, 0] );

              t_r.append(r_train[k, prop]);
              t_p.append(y_pred[1][j*BATCH_SIZE + k, prop, 0]);


    for j in range(len(test_gen)):
    
        a, b = test_gen[j]; 

        m_pred_test = a[3];
        w_test = a[1];

        y_test = b[0];
        r_test = b[1];

        for k in range(len(m_pred_test)):           

           if analyte not in m_pred_test[k]: continue;
           prop = np.where(m_pred_test[k] == analyte)[0][0]
 
           if len(set(w_test[k]).intersection(key)) > 0: continue;

           if y_test[k, prop] != MASKED_VALUE:
               qe.append( y_test[k, prop] );
               qr.append( y2_pred[0][j*BATCH_SIZE + k, prop, 0] );

               q_r.append(r_test[k, prop])
               q_p.append( y2_pred[1][j*BATCH_SIZE + k, prop, 0]);
    
    fpr, tpr, thresholds = roc_curve(q, v)
    roc_auc = auc(fpr, tpr)

    fp = open("stats/" + label + "-".join([t for t in ans]) + ".txt", "w");
    for i in range(len(fpr)):
       print(fpr[i], tpr[i], file=fp);
    fp.close();
    
    r1 = np.corrcoef(t_r, t_p) [0,1]
    r2 = np.corrcoef(q_r, q_p) [0,1]

    q_p = np.asarray(q_p);
    q_r = np.asarray(q_r);

    squared_differences = (q_p - q_r) ** 2
    rmse_test = np.sqrt(np.mean(squared_differences))

    return fpr, tpr, thresholds, qe, qr, roc_auc, r1*r1, r2*r2, rmse_test;

# calc for all augmentation variants
# i.g. predictin PSA only 
# PSA and Ferritin simultaneously
# PSA, Ferritin, and Uric acid 
# etc 

def calc_stats(analyte, name):
  
  keys =  get_variants(len(tests));
  for key in keys:
     
    if key[y_map[analyte]] != 1: continue; 
    
    nm = []
    ans = []

    for i in tests:
        if i.getId() != analyte and key[y_map[i.getId()]] == 1: 
            nm.append(i.getId());
            ans.append(i.getViewName());

    fpr, tpr, thresholds, qe, qr, roc_auc, r1, r2, rmse_test = calcRocs(analyte, set(nm), name, ans);

    m1 =  10
    m2 = -10

    ot1 = ot2 = 0.0
    for i in range(len(fpr)):
      r = tpr[i] - (1.0 - fpr[i])

      if r >= 0:
          if r <= m1:
             m1 = r
             ot1 = thresholds[i]
      else:
          if r > m2:
             m2 = r
             ot2 = thresholds[i]

    ot = (ot1 + ot2) / 2.0
 
    tp, tn, fp, fn = calc_confusion_matrix(qe, qr, ot);
  
    f1 = 2.0 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) !=0 else 0
    accuracy = (tp + tn) * 100.0/ ( tp + tn + fp + fn) if (tp + tn + fp + fn) !=0 else 0

    sensitivity = tp * 100.0 / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn * 100.0 / (tn + fp) if (tn + fp) > 0 else None

    print("AUC (", name + ": " + "-".join(ans), ") = ", roc_auc, "\n")
    print("Optimal threshold = ", ot)
    print("TP = ", tp)
    print("TN = ", tn)
    print("FP = ", fp)
    print("FN = ", fn, "\n")

    print("F1_score = ", f1)
    print("Accuracy = ", accuracy)
    print("Sensitivity = ", sensitivity)
    print("Specificity = ", specificity)

    print("");

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0 

    print("Positive Predictive Value = ", round(ppv,2) );
    print("Negative Predictive Value = ", round(npv,2) ) ;

    print("R-squared: ", r1, r2)
    print("RMSE: ", rmse_test)

    print("")

    if fp > 0 and fn > 0 and tp > 0 and tn > 0:
 
      dor =  tp * tn / fp / fn;

      sigma = 1.96 * math.sqrt(1.0/ tp + 1.0 / tn + 1.0 / fp + 1.0/ fn);
      low = math.exp( math.log(dor) - sigma);
      high = math.exp( math.log(dor) + sigma);

      print("DOR = ", round( dor, 1 ), " CI = [", round(low, 1), ", ", round(high,1) , "]")
      print("\n\n");

for i, analyte in enumerate(tests):
   calc_stats(analyte.getId(), analyte.getViewName());

print("Relax");

