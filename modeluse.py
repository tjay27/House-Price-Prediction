from joblib import dump, load
model = load('finalmodel.joblib')
print(model.predict([[-0.33006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24500176, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]]))