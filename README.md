# hw_ml
machine learning model for hasegawa wakatani turbulence.
prepare_data_vort.py post-processes outputs of a BOUT++ simulation. 
vort_nn.py produces local model for reynolds stress using a neural network for nonparametric regression; can be adapted for particle flux
test_model.py is used to produce plots of the learned model. (not up-to-date)
data is located in cleaned_data
