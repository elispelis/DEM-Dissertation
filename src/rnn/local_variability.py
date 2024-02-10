from edempy import Deck
import sys
sys.path.append("..")

from LaceyClass import LaceyMixingAnalyzer
from data_loader_rnn import RNNLoader
import pandas as pd
import numpy as np
import os

# sim_names = ["Rot_drum_mono.dem", "Rot_drum_binary_mixed.dem"]
# sim_name = sim_names[0]
# sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}"

# with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../Lacey_settings.txt")), 'r') as file:
#     preferences = file.readlines()
#     minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
#     maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
#     bins = np.array([int(i) for i in str(preferences[5]).split(',')])
#     cut_off = float(preferences[7])
#     plot = str(preferences[9])
#     file.close()
#     settings = True

# lacey = LaceyMixingAnalyzer()
# #rnn = RNNLoader(3,4,sim_path)