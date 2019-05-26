import numpy as np
from keras.models import Sequential
from keras.layers import Dense,GRU,Bidirectional,TimeDistributed
from keras import losses
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn as skl

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

data=np.array([]).reshape(0,512,3)
label=np.array([]).reshape(0,512)
dir_list=('kappa=0.75','kappa=1','kappa=1.25','kappa=1.5','kappa=1.75','kappa=2','kappa=2.25','kappa=2.5','kappa=2.75','kappa=3','beta=1','beta=1.5','beta=2','beta=2.5','beta=3','beta=4','beta=5','beta=6','beta=7','beta=8','beta=9','beta=10')#,'q=10','q=15','q=20','q=25','q=30')
for dir in dir_list:
    file=np.load(dir+'/data/cleaned_data_nonlocal.npz')
    newdata=np.stack((file['n'],file['vort'],file['ens']),axis=2)
    neg1=np.stack((np.fliplr(file['n']),np.fliplr(file['vort']),np.fliplr(file['ens'])),axis=2) #x-> -x, y-> -y
    neg2=np.stack((-np.fliplr(file['n']),-np.fliplr(file['vort']),np.fliplr(file['ens'])),axis=2) #x-> -x, phi-> -phi, n-> -n
    neg3=np.stack((-file['n'],-file['vort'],file['ens']),axis=2) #y-> -y, phi-> -phi, n-> -n
    #print(newdata.shape)
    data=np.concatenate((data,newdata,neg1,neg2,neg3),axis=0)
    #print(data.shape)
    newlabel=file['n_flux']
    #print(newlabel.shape)
    label=np.concatenate((label,-np.fliplr(newlabel),np.fliplr(newlabel),np.fliplr(newlabel),-newlabel),axis=0)

print(data.shape)

model=Sequential()

feat=['n','vort','ens']

ntrain=140000

#randomly partition data
shuffle_in_unison(data,label)
train_data=data[:ntrain,:]
val_data=data[ntrain+1:,:]
train_label=label[:ntrain]
val_label=label[ntrain+1:]

model.add(Bidirectional(GRU(512,input_shape=(512,3),return_sequences=True,dropout=0,recurrent_dropout=0),merge_mode='add'))
model.add(TimeDistributed(Dense(512)))
model.compile(loss='mean_squared_error', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='n_nonlocal.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(train_data, # Features
                      train_label, # Target vector
                      epochs=10000, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=1, # Print description after each epoch
                      batch_size=256, # Number of observations per batch
                      validation_data=(val_data, val_label)) # Data for evaluation
