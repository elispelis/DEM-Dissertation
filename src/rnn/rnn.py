import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import GRU, Dense, SimpleRNN

def rnn_arch(seq_length,num_features):
    # Define the RNN model
    model = Sequential()

    # Add a SimpleRNN layer with the specified number of units
    model.add(GRU(units=seq_length, activation='relu', input_shape=(seq_length, num_features)))

    # Add a Dense layer for the output with the appropriate number of units (equal to the number of features)
    model.add(Dense(units=num_features, activation='linear'))

    # Compile the model with an appropriate optimizer and loss function
    model.compile(optimizer='adam', loss='mse')  # Adjust optimizer and loss function as needed

    # Print the summary of the model architecture
    model.summary()

    return model


# model = Sequential()
# model.add(GRU(units=seq_length, input_shape =((seq_length, num_features)), return_sequences=True))
# model.add(Dense(units=3, activation="linear"))
# model.compile(optimizer='adam', loss='mse')

# # split a multivariate sequence into samples
# def split_sequences(sequences, n_steps):
# 	X, y = list(), list()
# 	for i in range(len(sequences)):
# 		# find the end of this pattern
# 		end_ix = i + n_steps
# 		# check if we are beyond the dataset
# 		if end_ix > len(sequences)-1:
# 			break
# 		# gather input and output parts of the pattern
# 		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
# 		X.append(seq_x)
# 		y.append(seq_y)
# 	return np.array(X), np.array(y)


def split_sequences(dataframe, n_steps):

    # Number of total timesteps
    total_timesteps = dataframe.shape[1]

    # Number of total samples
    total_samples = dataframe.shape[0]

    # Initialize empty lists to store input and output data
    x_data = []
    y_data = []

    # Iterate over samples to create input and output sequences
    for i in range(total_samples):
        for j in range(total_timesteps - n_steps):
            # Input sequence (previous timesteps)
            x_seq = dataframe[i, j:j+n_steps, :]
            
            # Output sequence (next timestep)
            y_seq = dataframe[i, j+n_steps, :]
            
            # Append to the lists
            x_data.append(x_seq)
            y_data.append(y_seq)

    return np.array(x_data), np.array(y_data) 


if __name__ == "__main__":

    df = pd.read_csv("../../model/400k_3_5_0.05s_adj.csv", index_col=0)

    num_features = 3
    num_timesteps = df.shape[1] // num_features
    num_particles = df.shape[0]
    seq_length = 15

    train_frac = 0.9
    train_size = int(train_frac*num_timesteps*num_features)
    train_df = df.iloc[:, :train_size]
    test_df = df.iloc[:, (num_timesteps-seq_length-1)*num_features:]

    train_df = train_df.values.reshape(-1, int(num_timesteps*train_frac), 3)
    test_df = test_df.values.reshape(-1, seq_length+1, 3)


    # Apply the split_sequences function to create input-output pairs
    X_train, y_train = split_sequences(train_df, seq_length)
    X_test, y_test = split_sequences(test_df, seq_length)

    model = rnn_arch(seq_length, num_features)

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
    # last_sequences = []

    # for i in np.arange(num_timesteps-seq_length-1, X_test.shape[0], num_timesteps-seq_length):
    #     last_sequences.append(X_test[i])

    # last_sequences = np.array(last_sequences)
    model.save(f"../../model/model_sl{seq_length}_tr{int(train_size/num_features)}_adj.h5")