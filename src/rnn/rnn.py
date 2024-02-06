import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import GRU, Dense, SimpleRNN

df = pd.read_csv("3_4_0.05s.csv", index_col=0)

num_timesteps = df.shape[1] // 3
num_particles = df.shape[0]
seq_length = 5

df = df.values.reshape(-1, num_timesteps, 3)

num_features = df.shape[2]


# Define the RNN model
model = Sequential()

# Add a SimpleRNN layer with the specified number of units
model.add(GRU(units=50, activation='relu', input_shape=(seq_length, num_features)))

# Add a Dense layer for the output with the appropriate number of units (equal to the number of features)
model.add(Dense(units=num_features, activation='linear'))

# Compile the model with an appropriate optimizer and loss function
model.compile(optimizer='adam', loss='mse')  # Adjust optimizer and loss function as needed

# Print the summary of the model architecture
model.summary()


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


train_size = int(len(df) * 0.8)  # 80% training, 20% testing
train, test = df[0:train_size], df[train_size:]

# Apply the split_sequences function to create input-output pairs
X_train, y_train = split_sequences(train, seq_length)
X_test, y_test = split_sequences(test, seq_length)

# # Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

last_sequences = []

for i in np.arange(num_timesteps-seq_length-1, X_test.shape[0], num_timesteps-seq_length):
    last_sequences.append(X_test[i])

last_sequences = np.array(last_sequences)