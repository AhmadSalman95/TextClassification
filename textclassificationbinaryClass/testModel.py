import os
import re
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers, losses
from tensorflow.python.keras.layers import TextVectorization
from tensorflow.keras import Sequential

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
max_features = 10000
sequence_length = 250
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
batch_size = 32
seed = 42

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_path1="checkpoint1/cp.ckpt"


def create_model():
    max_features = 10000
    sequence_length = 250
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    # model.load_weights(checkpoint_path)
    # model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])


    return model

model=create_model()
model.load_weights(checkpoint_path)
model.summary()
# for layer in model.layers:
#     wight=layer.get_weights()
#     print(wight)


export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
export_model.load_weights(checkpoint_path1)

# print("\n\n\nexport model wights")
# for layer in export_model.layers:
#     wights = layer.get_weights()
#     print(wights)


examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible...",
  "The movie is very bad",
  "its vrey bad, and the actors is terrible",
  "its good,every thing is good"
]

print(export_model.predict(examples))


# [[0.6584056 ]
#  [0.48382834]
#  [0.40073407]]
