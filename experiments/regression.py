
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
from tools import get_variants, prepareData

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

BATCH_SIZE=256

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
          mid, gra, b12, folic, bil_direct, 
          bil_indirect, bil_total, crea, urea, 
          pro, ldg, alb, crp, psa, 
          ferritin, uric, hba1c, ast, alt, chol, glu ]

#tests we want to predict 
tests = [source[-1]]

X = [];
Y = [];

sql = " is not null or ".join([ ("a." + analyte.getViewName()) for analyte in tests]) + " is not null";
sel = ", ".join([ ("a." + analyte.getViewName()) for analyte in source]);

y_map = {obj.getId() : i for i, obj in enumerate(tests)}
x_map = {obj.getId() : i for i, obj in enumerate(source)}

print(x_map);

for data in con.cursor().execute(f"""select {sel} from lab2 a 
                                     where ({sql}) and a.logdate <= '2025/04/01' and a.glu > 3.5 """):
        
   y = [ 0 ];       
   x = [ MASKED_VALUE ] * (len(source) - 1);
  
   for ind, value in enumerate(data):
       
      if source[ind].getId() != tests[0].getId():
         analyte_id = source[ind].getId();         
         v = source[ind].up(value);
         if v is not None: x[ x_map[analyte_id] ] = v;
      else:
          y[0] = source[ind].up(value);              

   X.append(x);
   Y.append(y);      

for analyte in source:
   print(analyte.getName(), analyte.getMinimum(), analyte.getMaximum());

with open("e3.pickle", 'wb') as file:
   pickle.dump(source, file);

print("Total: ", len(X));

l_in = Input(name='values', shape=(len(source)-1,))

l_1 = keras.layers.Dense(units=256, activation="relu")(l_in);
l_1 = keras.layers.Dropout(0.1)(l_1);

l_2 = keras.layers.Dense(units=64, activation="sigmoid")(l_1);
l_2 = keras.layers.Dropout(0.1)(l_2);

l_out = keras.layers.Dense(units=1, activation="sigmoid")(l_2);

model = Model(l_in, l_out);

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-3) , loss= "mse")
model.summary()


ids = np.arange(0, len(X), 1, dtype=int);
np.random.shuffle(ids);

train_size = (int)(len(ids) * 0.8)
test_size = len(ids) - train_size;

x_train = np.empty(shape=(train_size, len(source)-1));
y_train = np.empty(shape=(train_size, 1));


x_test = np.empty(shape=(test_size, len(source) -1));
y_test = np.empty(shape=(test_size, 1));

for i in range(0, train_size):
    for j in range(len(source)-1):
       x_train[i, j] = source[j].scale(X[ids[i]][j]);
    y_train[i, 0] = tests[0].scale(Y[ids[i]][0]);

j=0;
for i in range(train_size, len(ids)):
    for k in range(len(source)-1):
       x_test[j, k] = source[k].scale(X[ids[i]][k]);
    y_test[j,0] = tests[0].scale(Y[ids[i]][0]);
    j = j + 1;


print("Training sizes: ", len(x_train), len(y_train));
print("Test sizes: ", len(x_test), len(y_test));

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=100, verbose=1, mode="auto",
    baseline=None, restore_best_weights=True, start_from_epoch=0,
)

reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor = 0.99, patience = 20, verbose = 0, mode="auto",
    min_delta=0.00001, cooldown=0, min_lr= 1e-7)

model.fit(x = x_train, y= y_train, 
          shuffle = True, batch_size=BATCH_SIZE, epochs=1000, validation_split=0.2, 
          callbacks = [tensorboard_callback, early_stopping, reduce, tensorflow.keras.callbacks.TerminateOnNaN(), MessagerCallback()]) 

model.save_weights("ulm-base-regression.weights.h5", True);

y_pred = model.predict(x_train)
y2_pred = model.predict(x_test)

for i in range(y_train.shape[0]):
    y_train[i, 0] = tests[0].unscale(y_train[i, 0]);
    y_pred[i, 0] = tests[0].unscale(y_pred[i, 0]);

for i in range(y_test.shape[0]):
    y_test[i, 0] = tests[0].unscale(y_test[i, 0]);
    y2_pred[i, 0] = tests[0].unscale(y2_pred[i, 0]);

fp = open("stats/r.txt", "w");
for i in range(y2_pred.shape[0]):
   print(y_test[i, 0], y2_pred[i, 0], file=fp);


r1 = np.corrcoef(y_train[:, 0], y_pred[:, 0]) [0,1]
r2 = np.corrcoef(y_test[:, 0], y2_pred[:, 0]) [0,1]

print("R-squared: ", r1*r1, "/", r2*r2);
print("Relax");

