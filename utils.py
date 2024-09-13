import os
import pickle
# from constants import *

def load_model(ROOT_DIR_MODEL):
    print("MODEL LOADING STARTED............")
    model_path = os.path.join(ROOT_DIR_MODEL, 'model.h5')
    # model_path = r"{}".format(model_path)
    print(model_path)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def load_scaler(ROOT_DIR_SCALER):
    scaler_path = os.path.join(ROOT_DIR_SCALER, 'scaler')
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler


def load_encoders(ROOT_DIR_ENCODER):
    label_encoder_gender_path = os.path.join(ROOT_DIR_ENCODER, 'gender_label_encoder.pkl')
    with open(label_encoder_gender_path, 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    geo_one_hot_encoder_path = os.path.join(ROOT_DIR_ENCODER, 'geo_one_hot_encoder.pkl')
    with open(geo_one_hot_encoder_path, 'rb') as file:
        geo_one_hot_encoder = pickle.load(file)

    return label_encoder_gender,geo_one_hot_encoder