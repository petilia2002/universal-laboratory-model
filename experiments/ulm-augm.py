
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


config_object = ConfigParser()
config_object.read("../config.ini")

user_info = config_object["USERINFO"]
server_config = config_object["SERVERCONFIG"]

dsn = server_config["dsn"]
user = user_info["user"]
password = user_info["password"]

con = fdb.connect(dsn=dsn, user=user, password=password);

print(tensorflow.config.list_physical_devices('GPU'))

#enumerate all variants up to N 
#imitating addition in binary system, but from the left

# 1 0 0
# 0 1 0
# 1 1 0
# 0 0 1 
# 1 0 1 

# etc

def get_variants(n):
 
    v = [];

    a = [0 for i in range(n)]
    a  [0 ] = 1
  
    v.append(a[:]);  
    while True:

      i = -1;  
      for j in range(n):

          if a[j] == 1:
              a[j] = 0;
              i = j + 1;            
          else:               
              if i == -1 : i =j;            
              a[i] = 1;

              v.append(a[:]);
              break;
      if i >= n: break;
    return v;

# now use all analytes 

analytes = {}

# add gender and age as the do not have view names in our database 
idx = [34, 35]

for aid, name in con.cursor().execute("select id, view_name from analytes"):
    analytes[aid] = name;
    if name is not None: idx.append(aid);

#tests we want to predict 
tests = [1, 32, 41, 42, 30, 31];

NN = len(idx);

Limits = {};

for i in range(len(idx)):
    Limits[idx[i]] = [sys.float_info.max, -sys.float_info.max, 0];
for i in range(len(tests)):
    Limits[tests[i]] = [sys.float_info.max, -sys.float_info.max, 0];

X = [];
W = [];
Y = [];

sql = " is not null or ".join([ ("a." + analytes[aid]) for aid in tests]) + " is not null";

for gender, age, hgb, rbc, mcv, plt, wbc, neut, lymph, eo, baso, mono, \
    mid, gra, b12, folic, ast, alt, bil_direct, bil_indirect, bil_total, crea, urea, pro, ldg, alb, crp,\
    chol, glu, ferritin, uric, hba1c, psa  \
    in con.cursor().execute(f"""select  
                                      a.gender, a.age, a.hgb, a.rbc, a.mcv, a.plt, 
                                      a.wbc, a.neut, a.lymph, a.eo, a.baso, a.mono,
                                      a.mid, a.gra, a.b12, a.folic, a.ast, a.alt, 
                                      a.bil_direct, a.bil_indirect, a.bil_total, 
                                      a.crea, a.urea, a.pro, a.ldg, a.alb, a.crp, 
                                      a.chol, a.glu, a.ferritin, a.uric, a.hba1c, a.psa
                                from lab2 a 
                                where ({sql})
                                      and a.id <= 1671132 """):
        
   y = [MASKED_VALUE] * len(tests);       

   if ferritin is not None: 
       y[0] = 1 if ferritin <= 12 else 0;
   if uric is not None:
       y[1] = 1 if ((gender == 1 and uric >= 0.48) or (gender == 0 and uric >= 0.38)) else 0;
   if hba1c is not None:
       y[2] = 1 if hba1c >= 6.0 else 0;
   if psa is not None:
       y[3] = 1 if psa >= 4 else 0;
   if chol is not None:
       y[4] = 1 if chol >= 5.2 else 0   
   if glu is not None:
       y[5] = 1 if glu >= 7 else 0    

   x = [ 0 for i in range(NN) ];
   w = [ 0 for i in range(NN) ];
    
   z = -1;

   def add_value(analyte, ind):
       global z;

       if analyte is not None:
           v = float(analyte);
           if ind != 34:
               v = math.log(v);

           if Limits[ind][0] > v: Limits[ind][0] = v;
           if Limits[ind][1] < v: Limits[ind][1] = v;

           Limits[ind][2] = Limits[ind][2] + 1;

           z = z + 1;

           x[z] = v;
           w[z] = ind;
  
   add_value(gender, 34);
   add_value(age, 35);

   add_value(hgb, 19);
   add_value(rbc, 20); 
   add_value(mcv, 21);
   add_value(plt, 23);
   add_value(wbc, 22);
   add_value(neut, 24);
   add_value(lymph, 25);
   add_value(eo, 26);
   add_value(baso, 27);
   add_value(mono, 28);

   add_value(mid, 36);
   add_value(gra, 37),

   add_value(b12, 2);
   add_value(folic, 3);
   add_value(ast, 4);
   add_value(alt, 5);
   add_value(bil_direct, 6);
   add_value(bil_indirect, 7);
   add_value(bil_total, 8);
   add_value(crea, 9);
   add_value(urea, 10);
   add_value(pro, 11);
   add_value(ldg, 12);
   add_value(alb, 18);
   add_value(crp, 29);
   add_value(glu, 31);
   add_value(chol, 30);
 
   add_value(ferritin, 1);
   add_value(uric, 32);
   add_value(hba1c, 41);
   add_value(psa, 42);

   X.append(x);
   W.append(w);
   Y.append (y);    

for u in Limits:
   print(u, Limits[u]);

with open("e3.pickle", 'wb') as file:
   pickle.dump(Limits, file);

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

l_in = Input(name='values', shape=(NN,))
l_words = Input(name='analytes', shape=(NN,))
l_mask = Input(name='mask', shape=(NN,))

l_pred = Input(name='outcomes', shape=(len(tests),))

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

#we cannot flatten now
l_out_cbc = keras.layers.TimeDistributed( Dense(units= 1, activation = 'sigmoid'))(l_o);

def cbcLoss(y_true, y_pred):
   mask = K.cast_to_floatx(K.not_equal(y_true, MASKED_VALUE))
   y_pred = tensorflow.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon() );
   r = y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred);
   return -tensorflow.reduce_mean(r * mask );
   
model = Model([l_in, l_words, l_mask, l_pred], l_out_cbc);

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4) , loss= cbcLoss)
model.summary()

ids = np.arange(0, len(X), 1, dtype=int);
np.random.shuffle(ids);

train_size = (int)(len(ids) * 0.8)

#input
x_train = []
w_train = []
m_train = []

#output
y_train = []
m_pred_train = []

for i in range(train_size):  

    a = [0] * NN

    for j in range(NN):

       v = X[ids[i]][j];
       w = W[ids[i]][j];

       if w == 0: break; 
       
       y_min, y_max = Limits[w][0], Limits[w][1];
 
       add = 0.01 * (y_max - y_min);
       y_max = y_max + add;
       y_min = y_min - add;

       a[j] = 0.9 - 0.8 * (y_max - v) / (y_max - y_min);

    y_t = Y[ids[i]];
    w_t = W[ids[i]];

    words = [];
    for j in range(len(tests)):
        if y_t[j] != MASKED_VALUE: words.append(tests[j]);

    #augment input
    for p in get_variants(len(words)):

       b = [0] * len(tests)
       c = [MASKED_VALUE] * len(tests)

       ind = 0
       dels = []

       for j in range(len(p)):
           if p[j] == 1:
              b[ind] = words[j]
              c[ind] = y_t [ tests.index(words[j]) ]
              ind = ind + 1  
              dels.append(words[j])

       t = [0] * NN
       m = [0] * NN

       w = w_t[:]      

       for d in dels:
           w [ w.index(d) ] = 0;

       ind = 0;
       for j in range(NN):
           if w[j] != 0:
               t [ind] = a [j]
               m [ind] = w [j]
               ind = ind + 1
   
       m_t = [0] * NN
       for j in range(NN):
           if m[j] > 0: m_t[j] = 1;       

       x_train.append(t)
       w_train.append(m)
       m_train.append(m_t)

       y_train.append(c)
       m_pred_train.append(b)


x_train = np.asarray(x_train);
w_train = np.asarray(w_train);
m_train = np.asarray(m_train);

y_train = np.asarray(y_train);
m_pred_train = np.asarray(m_pred_train);

test_size = len(ids) - train_size;

x_test = [];
w_test = [];
m_test = [];

y_test = [];
m_pred_test = [];

for i in range(train_size, len(ids)):

    a = [0] * NN
    
    for k in range(NN):

       v = X[ids[i]][k];
       w = W[ids[i]][k];

       if w == 0: break; 
       
       y_min, y_max = Limits[w][0], Limits[w][1];
 
       add = 0.01 * (y_max - y_min);
       y_max = y_max + add;
       y_min = y_min - add;

       a[k] = 0.9 - 0.8 * (y_max - v) / (y_max - y_min);

    y_t = Y[ids[i]];
    w_t = W[ids[i]];
     
    words = [];
    for j in range(len(tests)):
        if y_t[j] != MASKED_VALUE: words.append(tests[j]);

    for p in get_variants(len(words)):

       b = [0] * len(tests)
       c = [MASKED_VALUE] * len(tests)

       ind = 0
       dels = []

       for j in range(len(p)):
           if p[j] == 1:
              b[ind] = words[j]
              c[ind] = y_t [ tests.index(words[j]) ]
              ind = ind + 1  
              dels.append(words[j])

       t = [0] * NN
       m = [0] * NN

       w = w_t[:]      

       for d in dels:
           w [ w.index(d) ] = 0;

       ind = 0;
       for j in range(NN):
           if w[j] != 0:
               t [ind] = a [j]
               m [ind] = w [j]
               ind = ind + 1
   
       m_t = [0] * NN
       for j in range(NN):
           if m[j] > 0: m_t[j] = 1;       

       x_test.append(t)
       w_test.append(m)
       m_test.append(m_t)

       y_test.append(c)
       m_pred_test.append(b)

print("Training sizes: ", len(x_train), len(y_train));
print("Test sizes: ", len(x_test), len(y_test));

x_test = np.asarray(x_test);
w_test = np.asarray(w_test);
m_test = np.asarray(m_test);

y_test = np.asarray(y_test);
m_pred_test = np.asarray(m_pred_test);

#model.load_weights("ulm-1.weights.h5");

# Output must depend on the order of keys  
# To check if output depends on the order of keys 

'''
x1 = m_pred_train[0:2][:];

x1[0, 0] = 1
x1[0, 1] = 0

x1[1, 0] = 32
x1[1, 1] = 1

i1 = x_train[0:2][:]
i1[1] = i1[0]

i2 = w_train[0:2][:]
i2[1] = i2[0]

i3 = m_train[0:2][:]
i3[1] = i3[0]

print(x1);

model.load_weights("ulm-1.weights.h5");

a = model([i1, i2, i3, x1]);

print("pred")
print(a)

sys.exit(0);
'''

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)

reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor = 0.9,
    patience = 20,
    verbose = 0,
    mode="auto",
    min_delta=0.00001,
    cooldown=0,
    min_lr= 1e-7)

model.fit(x = [x_train, w_train, m_train, m_pred_train], y= y_train, 
          shuffle = True, batch_size=16, epochs=1, validation_split=0.2, 
          callbacks = [tensorboard_callback, early_stopping, reduce, tensorflow.keras.callbacks.TerminateOnNaN(), MessagerCallback()]) 

model.save_weights("ulm-1.weights.h5", True);

y_pred = model.predict([x_train, w_train, m_train, m_pred_train])
y2_pred = model.predict([x_test, w_test, m_test, m_pred_test])

def calcRocs(analyte, key):
    
    q = [];
    v = [];
    
    qe = [];
    qr = [];
    
    for i in range(len(y_train)):

        if analyte not in m_pred_train[i]: continue;
        prop = np.where(m_pred_train[i] == analyte)[0][0]
       
        if len(set(w_train[i]).intersection(key)) > 0: continue;

        if y_train[i, prop] != MASKED_VALUE:
            q.append( y_train[i, prop] );
            v.append( y_pred[i, prop] );
    
    for i in range(len(y2_pred)):

        if analyte not in m_pred_test[i]: continue;
        prop = np.where(m_pred_test[i] == analyte)[0][0]
 
        if len(set(w_test[i]).intersection(key)) > 0: continue;

        if y_test[i, prop] != MASKED_VALUE:
            qe.append( y_test[i, prop] );
            qr.append( y2_pred[i, prop] );
    
    fpr, tpr, thresholds = roc_curve(q, v)
    roc_auc = auc(fpr, tpr)

    label = "";
    
    if prop in analytes: label = analytes[prop]
    
    fp = open("stats/" + label + "-".join([str(t) for t in key]) + ".txt", "w");
    for i in range(len(fpr)):
       print(fpr[i], tpr[i], file=fp);
    fp.close();
        
    return fpr, tpr, thresholds, qe, qr, roc_auc;



#calc for all augmentation variants
# i.g. predictin PSA only 
# PSA and Ferriting simultaneously
# PSA, Ferritin, and Uric acid 
# etc 

def calc_stats(analyte, name):
  
  keys =  get_variants(len(tests));
  for key in keys:
     
    if key[tests.index(analyte)] != 1: continue; 
    
    nm = []
    ans = []

    for i in tests:
        if i != analyte and key[tests.index(i)] == 1: 
            nm.append(i);
            ans.append(analytes[i]);

    fpr, tpr, thresholds, qe, qr, roc_auc = calcRocs(analyte, set(nm));

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

    print("")

    if fp > 0 and fn > 0 and tp > 0 and tn > 0:
 
      dor =  tp * tn / fp / fn;

      sigma = 1.96 * math.sqrt(1.0/ tp + 1.0 / tn + 1.0 / fp + 1.0/ fn);
      low = math.exp( math.log(dor) - sigma);
      high = math.exp( math.log(dor) + sigma);

      print("DOR = ", round( dor, 1 ), " CI = [", round(low, 1), ", ", round(high,1) , "]")
      print("\n\n");

for i in tests:
   calc_stats(i, analytes[i]);

print("Relax");

