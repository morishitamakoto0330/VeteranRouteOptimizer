import tensorflow as tf

# (10, 10)�̓��̓f�[�^�쐬
# (100, 1)�̋��t�f�[�^�쐬


"""
input_shape = (4, 10, 10, 1)
x = tf.random.normal(input_shape)
y = tf.keras.Sequential()

y.add(tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_shape[1:]))
print(y(x).shape)
y.add(tf.keras.layers.Flatten())
print(y(x).shape)
y.add(tf.keras.layers.Dense(100))
print(y(x).shape)
"""





