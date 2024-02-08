import tensorflow as tf
import numpy as np
import pandas as pd
from edempy import Deck
import extrapolation

sim_names = ["Rot_drum_mono.dem", "Rot_drum_binary_mixed.dem"]
sim_name = sim_names[0]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}"
model_path = "../model/model_sl10_tr144.h5"
data_path = "../model/3_4_0.05s.csv"


extrap = extrapolation()

model = tf.keras.models.load_model(model_path)
df = pd.read_csv(data_path, index_col=0)

num_features = 3
num_timesteps = df.shape[1] // num_features
num_particles = df.shape[0]
seq_length = 10

last_seq = df.iloc[:, (num_timesteps-seq_length)*num_features:]

last_seq = last_seq.values.reshape(-1, seq_length, 3)

t_rnn = 0.05
extrap_time = t_rnn


for i in range(int(extrap_time/t_rnn)):
    pred_timestep = model.predict(last_seq)
    last_seq = last_seq[:, 1:, :]
    last_seq = np.concatenate((last_seq, pred_timestep[:, np.newaxis, :]), axis=1)