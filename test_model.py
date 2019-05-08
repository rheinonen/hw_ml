import numpy as np
import xgboost as xgb
import sklearn as skl
import time
from matplotlib import pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense
from keras.models import load_model
#from keras import regularizers
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#import sklearn as skl

#models = [xgb.Booster({'nthread': 2}) for i in range(0,5)]
#for i in range(0,5):
#    models[i].load_model(str(i)+'.model')
n_models=10

# diagonal component of density flux
# enslist=(0.1, 10, 20, 30, 40,50)
# plt.xlabel('grad n')
# plt.ylabel('n flux (at grad u=0)')
#
# for ens in enslist:
#     average_flux=np.zeros((1601,))
#     predict_data=np.transpose([np.ones(1601)*ens,np.ones(1601), [i/100. for i in range(-800,801)],np.zeros(1601),np.zeros(1601)])
#     for i in range(0,n_models):
#         model = load_model("n_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-800,801)],average_flux)
#     time.sleep(1)
#     print(str(ens))
# plt.legend(['ens='+str(ens) for ens in enslist])
# plt.show()

#diagonal component of density flux---dependence on vorticity
# vortlist=(-6,-4,-2,0, 2, 4, 6)
# plt.xlabel('grad n')
# plt.ylabel('n flux')
# for vort in vortlist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*10,np.ones(1001)*vort, [i/100. for i in range(-500,501)],np.zeros(1001),np.zeros(1001)])
#     for i in range(0,5):
#         model = load_model("n_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(str(vort))
# plt.legend(['vort='+str(vort) for vort in vortlist])
# plt.show()

#density flux---dependence on both gradients
gradlist=(-4,-3,-2,-1, 0,1,2,3,4)

plt.xlabel('grad vort')
plt.ylabel('n flux')
for grad in gradlist:
    average_flux=np.zeros((501,))
    predict_data=np.transpose([np.ones(501)*10,np.ones(501), np.ones(501)*grad,[i/50. for i in range(-250,251)],np.zeros(501)])
    for i in range(0,n_models):
        model = load_model("n_"+str(i)+".h5")
        predicted_flux=model.predict(predict_data)
        average_flux=average_flux+predicted_flux
    average_flux=average_flux/n_models
    plt.plot([i/50. for i in range(-250,251)],average_flux)
    time.sleep(1)
    print(grad)
plt.legend(['grad n='+str(grad) for grad in gradlist])
plt.show()

#off-diagonal component of density flux ---dependence on enstrophy
enslist=(0.1, 10, 20, 30, 40,50)

# plt.xlabel('grad vort')
# plt.ylabel('n flux (at grad n=0)')
# for ens in enslist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*ens,np.ones(1001),np.zeros(1001),[i/100. for i in range(-500,501)],np.zeros(1001)])
#     for i in (3,):# range(0,n_models):
#         model = load_model("n_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     #average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(ens)
# plt.legend(['ens='+str(ens) for ens in enslist])
# plt.show()

#off-diagonal component of density flux --- dependence on vorticity
# vortlist=(-6, -4, -2, 0, 2,4,6)
# plt.xlabel('grad vort')
# plt.ylabel('n flux (at grad n=0)')
# for vort in vortlist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*10,np.ones(1001)*vort,np.zeros(1001),[i/100. for i in range(-500,501)],np.zeros(1001)])
#     for i in range(0,n_models):
#         model = load_model("n_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(vort)
# plt.legend(['vort='+str(vort) for vort in vortlist])

# plt.show()
#
#diagonal component of vorticity flux
# enslist=(0.1, 10, 20, 30, 40,50)
# plt.xlabel('grad vort')
# plt.ylabel('vort flux (at grad n=0)')
# for ens in enslist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*ens,np.ones(1001),np.zeros(1001),[i/100. for i in range(-500,501)],np.zeros(1001)])
#     for i in range(0,n_models):
#         model = load_model("vort_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(ens)
# plt.legend(['ens='+str(ens) for ens in enslist])
#
# plt.show()

#vort flux---dependence on both gradients
# gradlist=(-4,-3,-2,-1, 0,1,2,3,4)
#
# plt.xlabel('grad vort')
# plt.ylabel('vort flux')
# for grad in gradlist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*10,2*np.ones(1001), np.ones(1001)*grad,[i/100. for i in range(-500,501)],np.zeros(1001)])
#     for i in range(0,n_models):
#         model = load_model("vort_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(grad)
# plt.legend(['grad n='+str(grad) for grad in gradlist])
# plt.show()

#off-diagonal component of vorticity flux
# enslist=(0.1, 5,10,15,20)
# plt.xlabel('grad n')
# plt.ylabel('vort flux (at grad vort=0)')
# for ens in enslist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*ens,np.ones(1001),[i/100. for i in range(-500,501)],np.zeros(1001),np.zeros(1001)])
#     for i in range(5,):#range(0,n_models):
#         model = load_model("vort_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     #average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(ens)
# plt.legend(['ens='+str(ens) for ens in enslist])
#
# plt.show()

#enstrophy flux dependence on enstrophy
# enslist=(0.1,5, 10, 15, 20)
#
# plt.xlabel('grad ens')
# plt.ylabel('ens flux')
# for ens in enslist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*ens,np.zeros(1001),np.zeros(1001),np.zeros(1001),[i/100. for i in range(-500,501)]])
#     for i in range(0,n_models):
#         model = load_model("ens_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(ens)
# plt.legend(['ens='+str(ens) for ens in enslist])
# plt.show()

#enstrophy flux dependence on vorticity
# vortlist=(-2,-1,0,1,2)
#
# plt.xlabel('grad ens')
# plt.ylabel('ens flux')
# for vort in vortlist:
#     average_flux=np.zeros((1001,))
#     predict_data=np.transpose([np.ones(1001)*10,vort*np.ones(1001),np.zeros(1001),np.zeros(1001),[i/100. for i in range(-500,501)]])
#     for i in range(0,n_models):
#         model = load_model("ens_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     average_flux=average_flux/n_models
#     plt.plot([i/100. for i in range(-500,501)],average_flux)
#     time.sleep(1)
#     print(vort)
# plt.legend(['vort='+str(vort) for vort in vortlist])
# plt.show()

# vorticity flux fluctuations
# vortlist=(-2,-1,0,1,2)
#
# plt.xlabel('grad ens')
# plt.ylabel('ens flux')
# for vort in vortlist:
#     average_flux=np.zeros((51,))
#     predict_data=np.transpose([[i for i in range(0,51)],vort*np.ones(51)])
#     for i in range(0,):
#         model = load_model("vort_std_"+str(i)+".h5")
#         predicted_flux=model.predict(predict_data)
#         average_flux=average_flux+predicted_flux
#     #average_flux=average_flux/n_models
#     plt.plot([i for i in range(0,51)],average_flux)
#     time.sleep(1)
#     print(vort)
# plt.legend(['vort='+str(vort) for vort in vortlist])
# plt.show()
