import sys 
import math
import numpy as np
import keras

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

# Preparation augmented dataset.
# Generator version.

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, start, end, ids, X, W, Y, source, tests, x_map, y_map, mask_value, batch_size):
        super().__init__( )

        self.ids = ids[start : end]        

        self.X = X
        self.W = W
        self.Y = Y
        self.source = source
        self.tests = tests
        self.x_map = x_map

        self.y_map = y_map
        self.mask_value = mask_value
        self.batch_size = batch_size

        self.shuffle = True
        self.total = 0
         
        for i in self.ids:        
           cnt = 0  
           for j in range(len(self.tests)):
              if Y[i][j] != self.mask_value: cnt = cnt + 1;
           self.total = self.total + int(math.pow(2, cnt)) - 1;

        self.batches = []
        self.on_epoch_end()

    def noShuffle(self):
        self.shuffle = False
        self.on_epoch_end()

    def __len__(self):    
        return len(self.batches);

    def __getitem__(self, index):

        sid = self.batches[index][0];
        vid = self.batches[index][1];

        #input
        x_d = [] 
        w_d = []
        m_d = []

        #output
        y_d = []
        m_pt = []    
        r_d = []    
    
        for i in range(sid, len(self.ids)):

          a = [0] * len(self.source)
          for j in range(len(self.source)):

             v = self.X[self.ids[i]][j];
             w = self.W[self.ids[i]][j];

             if w == 0: break;      
             a[j] = self.source[ self.x_map[w] ].scale(v);

          y_t = self.Y[self.ids[i]];
          w_t = self.W[self.ids[i]];

          words = [];
          for j in range(len(self.tests)):
             if y_t[j] != self.mask_value: words.append(self.tests[j].getId());           

          #augment input
          ca = -1;

          for p in get_variants(len(words)):
              ca = ca + 1;

              if vid != -1 and ca < vid - 1 and i == sid : continue;             
              
              b = [0] * len(self.tests)
              c = [self.mask_value] * len(self.tests)
              e = [0] * len(self.tests)

              ind = 0
              dels = []

              for j in range(len(p)):
                  if p[j] == 1:
                     b[ind] = words[j]
                     c[ind] = y_t [ self.y_map[words[j]] ]

                     ind_source = w_t.index(words[j])                
                     e[ind] = self.source[self.x_map[words[j]]].scale(self.X[self.ids[i]][ind_source])

                     ind = ind + 1  
                     dels.append(words[j])

              t = [0] * len(self.source)
              m = [0] * len(self.source)

              w = w_t[:]      

              for d in dels:
                  w [ w.index(d) ] = 0;
  
              ind = 0;
              for j in range(len(self.source)):
                  if w[j] != 0:
                      t [ind] = a [j]
                      m [ind] = w [j]
                      ind = ind + 1
   
              m_t = [0] * len(self.source)
              for j in range(len(self.source)):
                  if m[j] > 0: m_t[j] = 1;       

              x_d.append(t)
              w_d.append(m)
              m_d.append(m_t)

              y_d.append(c)
              m_pt.append(b)
              r_d.append(e)

              #full sized batch
              if len(x_d) >= self.batch_size:
                
                  x_d = np.asarray(x_d, dtype="float32");
                  w_d = np.asarray(w_d, dtype="int8");
                  m_d = np.asarray(m_d, dtype="int8");

                  y_d = np.asarray(y_d, dtype="int8");
                  m_pt = np.asarray(m_pt, dtype="int8");
                  r_d = np.asarray(r_d, dtype="float32");

                  return (x_d, w_d, m_d, m_pt), (y_d, r_d)

        #last small batch
        x_d = np.asarray(x_d, dtype="float32");
        w_d = np.asarray(w_d, dtype="int8");
        m_d = np.asarray(m_d, dtype="int8");

        y_d = np.asarray(y_d, dtype="int8");
        m_pt = np.asarray(m_pt, dtype="int8");
        r_d = np.asarray(r_d, dtype="float32");

        return (x_d, w_d, m_d, m_pt), (y_d, r_d)
 

    def on_epoch_end(self):

        if self.shuffle: np.random.shuffle(self.ids)      

        self.batches = [];

        batch_cnt = 0
        batch = 0

        it = [0, -1];
        self.batches.append(it[:])

        for k, i in enumerate(self.ids):

            cnt = 0;
            for j in range(len(self.tests)):
              if self.Y[i][j] != self.mask_value: cnt = cnt + 1;
              
            cnt = int(math.pow(2, cnt)) - 1
            batch_cnt = batch_cnt + cnt;

            if batch_cnt > self.batch_size:

               it[0] = k;
               it[1] = cnt - (batch_cnt - self.batch_size) + 1; 

               self.batches.append(it[:])

               #left from this batch add to the next
               batch_cnt = cnt - it[1] + 1 
               
            if batch_cnt == self.batch_size:
               batch_cnt = 0
               if k < len(self.ids) - 1:
                   it[0] = k + 1
                   it[1] = -1
                   self.batches.append(it[:])


