import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import losses
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
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

#import outputs of post_process.py for several simulations. kappa=constant density gradient drive, beta=curved density gradient drive
# neg1,neg2,neg3 enforce symmetry constraints of exact equations under sign inversion
num_inputs=5
data=np.array([]).reshape(0,num_inputs)
label=[]
#dir_list=('q=3_kappa=1','q=3_kappa=1.5','q=3_kappa=2','q=3_kappa=2.5','q=3_kappa=3','q=1_kappa=1','q=1_kappa=1.5','q=1_kappa=2','q=1_kappa=2.5','q=1_kappa=3','q=2_kappa=1','q=2_kappa=1.5','q=2_kappa=2','q=2_kappa=2.5','q=2_kappa=3',
dir_list=('kappa=0.75','kappa=1','kappa=1.25','kappa=1.5','kappa=1.75','kappa=2','kappa=2.25','kappa=2.5','kappa=2.75','kappa=3','beta=1','beta=1.5','beta=2','beta=2.5','beta=3','beta=4','beta=5')#'beta=6','beta=7','beta=8','beta=9','beta=10')#,'q=10','q=15','q=20','q=25','q=30')
for dir in dir_list:
    try:    
        file=np.load(dir+'/data/cleaned_data_all.npz')
        newdata=np.stack((file['vort_xx'],file['vort_x'],file['vort'],file['n_x'],file['ens']),axis=1)#,file['vort_x'],file['n_xx'],file['vort_xx']),axis=1)
        neg1=np.stack((file['vort_xx'],-file['vort_x'],file['vort'],-file['n_x'],file['ens']),axis=1)#,-file['vort_x'],file['n_xx'],file['vort_xx']),axis=1) #x-> -x, y-> -y
        neg2=np.stack((-file['vort_xx'],file['vort_x'],-file['vort'],file['n_x'],file['ens']),axis=1)#,file['vort_x'],-file['n_xx'],-file['vort_xx']),axis=1) #x-> -x, phi-> -phi, n-> -n
        neg3=np.stack((-file['vort_xx'],-file['vort_x'],-file['vort'],-file['n_x'],file['ens']),axis=1)#,-file['vort_x'],-file['n_xx'],-file['vort_xx']),axis=1) #y-> -y, phi-> -phi, n-> -n
    #print(newdata.shape)
        data=np.concatenate((data,newdata,neg1,neg2,neg3),axis=0)
    #print(data.shape)
        newlabel=file['n_flux']
    #print(newlabel.shape)
        label=np.concatenate((label,newlabel,-newlabel,newlabel,-newlabel),axis=0)
    except:
        print("problem loading "+dir)
print(data.shape)

#these lines are for testing a trained model on an unseen data set

# test_list=('kappa=1.75','kappa=2')
# test_data=np.array([]).reshape(0,5)
# test_label=[]
# for dir in test_list:
#     file=np.load(dir+'/data/cleaned_data.npz')
#     newdata=np.stack((file['ens'],file['vort'],file['n_x'],file['vort_x'],file['ens_x']),axis=1)
#     test_data=np.concatenate((test_data,newdata),axis=0)
#     newlabel=file['vort_flux']
#     test_label=np.concatenate((test_label,newlabel),axis=0)

#train model several times to make sure independent of training/validation partition
n_models=5
ntrain=int(data.shape[0]*0.8)
models = [Sequential() for i in range(0,n_models)]
feat=['vort_x','vort','n_x','n_xx']
for i in range(0,n_models):
    shuffle_in_unison(data,label)
    train_data=data[:ntrain,:]
    val_data=data[ntrain+1:,:]
    train_label=label[:ntrain]
    val_label=label[ntrain+1:]


    models[i].add(Dense(units=8,use_bias=True,input_dim=num_inputs,kernel_regularizer=regularizers.l2(0.00001)))
    models[i].add(ELU(alpha=1))
    models[i].add(BatchNormalization())
    #models[i].add(Dropout(0.2))
    models[i].add(Dense(units=8,use_bias=True,kernel_regularizer=regularizers.l2(0.00001)))
    models[i].add(ELU(alpha=1))
    models[i].add(BatchNormalization())
    #models[i].add(Dropout(0.5))
    models[i].add(Dense(units=8,use_bias=True,kernel_regularizer=regularizers.l2(0.00001)))
    models[i].add(ELU(alpha=1))
    models[i].add(BatchNormalization())
    #models[i].add(Dropout(0.5))
    models[i].add(Dense(units=1))

    #logcosh makes the model more robust to large fluctuations
    models[i].compile(loss='logcosh', optimizer='adam')

    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(filepath="particle2_"+str(i)+'.h5', monitor='val_loss', save_best_only=True)]
    history = models[i].fit(train_data, # Features
                          train_label, # Target vector
                          epochs=100, # Number of epochs
                          callbacks=callbacks, # Early stopping
                          verbose=1, # Print description after each epoch
                          batch_size=256, # Number of observations per batch
                          validation_data=(val_data, val_label)) # Data for evaluation

    #test=models[i].evaluate(x=test_data,y=test_label,verbose=1)
    #print(test)
