pip install --upgrade pip
pip uninstall protobuf
pip install protobuf==3.19.0

def cnn_lstm_model(input_shape):
    
    model = Sequential()
    model.add(Conv1D(32,kernel_size=1,activation='relu',input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(units=32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    return model

input_shape = (n_timesteps,n_features)
model_2 = cnn_lstm_model(input_shape)

model_2.summary()
