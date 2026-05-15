import pickle

with open(r"C:\Users\KARISHMA\Desktop\deep learning\Depth_Estimation\training_outputs\training_11-24april resnet+unet 50k dataset 2.0\all_results_v8.pkl", "rb") as f:
    data = pickle.load(f)

print(data)