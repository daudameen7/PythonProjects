import pickle

loaded_model = pickle.load(open('Week8Model.pkl', 'rb'))  # loads the saved model data

print(loaded_model.intercept_)  # bias of data
print(loaded_model.coef_)  # weight of data