import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import losses
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn as skl
#import h5py
#from matplotlib import pyplot as plt
#import pickle as pkl

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

data=np.array([]).reshape(0,5)
label=[]
dir_list=('kappa=0.75','kappa=1','kappa=1.25','kappa=1.5','kappa=1.75','kappa=2','kappa=2.25','kappa=2.5','kappa=2.75','kappa=3','beta=1','beta=1.5','beta=2','beta=2.5','beta=3','beta=4','beta=5','beta=6','beta=7','beta=8','beta=9','beta=10')#,'q=10','q=15','q=20','q=25','q=30')
for dir in dir_list:
    file=np.load(dir+'/data/cleaned_data.npz')
    newdata=np.stack((file['ens'],file['vort'],file['n_x'],file['vort_x'],file['ens_x']),axis=1)
    neg1=np.stack((file['ens'],file['vort'],-file['n_x'],-file['vort_x'],-file['ens_x']),axis=1)
    neg2=np.stack((file['ens'],-file['vort'],file['n_x'],file['vort_x'],-file['ens_x']),axis=1)
    neg3=np.stack((file['ens'],-file['vort'],-file['n_x'],-file['vort_x'],file['ens_x']),axis=1)
    #print(newdata.shape)
    data=np.concatenate((data,newdata,neg1,neg2,neg3),axis=0)
    #print(data.shape)
    newlabel=file['n_flux']
    #print(newlabel.shape)
    label=np.concatenate((label,newlabel,-newlabel,newlabel,-newlabel),axis=0)

print(data.shape)
#shuffle_in_unison(data,label)
#w=-np.log(pdf(label))
#print(np.shape(w))

#train_weight=w[20001:]
#test_weight=w[:20000]

n_models=10
models = [Sequential() for i in range(0,n_models)]
feat=['ens','vort','n_x','vort_x','ens_x']
for i in range(0,n_models):
    shuffle_in_unison(data,label)
    train_data=data[:2000000,:]
    test_data=data[2000001:,:]
    train_label=label[:2000000]
    test_label=label[2000001:]

    models[i].add(Dense(units=10, input_dim=5,kernel_regularizer=regularizers.l2(0.1)))
    #models[i].add(BatchNormalization())
    models[i].add(ELU(alpha=1))
    models[i].add(BatchNormalization())
    #models[i].add(Dropout(0.5))
    models[i].add(Dense(units=10,kernel_regularizer=regularizers.l2(0.1)))
    #models[i].add(BatchNormalization())
    models[i].add(ELU(alpha=1))
    models[i].add(BatchNormalization())
    #models[i].add(Dropout(0.5))
    models[i].add(Dense(units=10, kernel_regularizer=regularizers.l2(0.1)))
    #models[i].add(BatchNormalization())
    models[i].add(ELU(alpha=1))
    models[i].add(BatchNormalization())
    #models[i].add(Dropout(0.5))
    models[i].add(Dense(units=1))
    models[i].compile(loss='logcosh', optimizer='adam')

    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(filepath="n_"+str(i)+'.h5', monitor='val_loss', save_best_only=True)]
    history = models[i].fit(train_data, # Features
                          train_label, # Target vector
                          epochs=1000, # Number of epochs
                          callbacks=callbacks, # Early stopping
                          verbose=1, # Print description after each epoch
                          batch_size=256, # Number of observations per batch
                          validation_data=(test_data, test_label)) # Data for evaluation
    #models[i].save("n_"+str(i)+'_nn.h5')

#enslist=(0.1, 10,20, 30, 40)
#av=np.zeros(3001)
#for ens in enslist:
#    predict_data=np.transpose([np.ones(1001)*ens,np.ones(1001), [i/100. for i in range(-500,501)],np.zeros(1001),np.zeros(1001)])
#    for i in range(0,5):
    #model.fit(train_data,train_label,sample_weight=train_weight,eval_set=[(test_data, test_label)],sample_weight_eval_set=[test_weight],early_stopping_rounds=50)
#        predicted_flux=models[i].predict(predict_data)
        #plt.scatter([i/100. for i in range(0,1501)],predicted_flux)
#        av=av+predicted_flux
#    av=av/5.
#    plt.plot([i/100. for i in range(-500,501)],predicted_flux)
#plt.legend(['ens='+str(ens) for ens in enslist])
#plt.show()
