import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, SpatialDropout1D, Flatten
from tensorflow.keras.layers import InputLayer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# read dataset
df = pd.read_csv('Dataset/drink_level_log.csv')

# Extract and preprocess timestamp features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute

# Select features and target columns
target = ['cof_level', 'milk_level', 'con_level', 'Year', 'Month', 'Day', 'Hour', 'Minute']

# Dataset preprocessing: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[target])


# create time sequence data
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0:3])  # predict the first 3 characteristic
    return np.array(X), np.array(y)


time_step = 50
X, y = create_dataset(scaled_data, time_step)

# divide into training set and test set
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# create TCN Model
model = Sequential()

# input layer
model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))

# TCN layer
model.add(Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SpatialDropout1D(0.2))

model.add(Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SpatialDropout1D(0.2))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(y_train.shape[1]))

# compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# train Model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Predict on the test data
y_pred_prob = model.predict(X_test)

# Convert predictions and true values to binary using a threshold
threshold = 0.50
y_pred_binary = (y_pred_prob > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

# Flatten the arrays for metric calculation
y_pred_flat = y_pred_binary.flatten()
y_test_flat = y_test_binary.flatten()

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test_flat, y_pred_flat)
precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_flat).ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
# Calculate False Discovery Rate (FDR)
fdr = fp / (fp + tp) if (fp + tp) > 0 else 0

# Print additional metrics
print(f'False Positive Rate (FPR): {fpr:.2f}')
print(f'False Discovery Rate (FDR): {fdr:.2f}')
