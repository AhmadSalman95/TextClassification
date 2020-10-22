import tensorflow as tf
from tensorflow.keras import models, layers, losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import os
import string
import re


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 100000
sequence_length = 1000
batch_size = 32
seed = 42
className=['csharp','java','javascript','python']
vectorize_layer = TextVectorization(max_tokens=max_features, standardize=custom_standardization, output_mode='int',
                                    output_sequence_length=sequence_length)
checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_pathExportModel = "checkpointExportModel/cp.ckpt"


def create_model():
    max_features = 100000
    sequence_length = 1000
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.5),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')])

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.SparseCategoricalAccuracy())
    return model


model = create_model()
model.load_weights(checkpoint_path)
model.summary()

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
export_model.load_weights(checkpoint_pathExportModel)

examples = [
    "how do i extract keys from a dict into a list?",
    "debug public static void main(string[] args){...}"
]

def get_string_labels(batch):
    predicted_int_labels= tf.argmax(batch,axis=1)
    predicted_labels=tf.gather(className,predicted_int_labels)
    return predicted_labels

prediction = export_model.predict(examples)
prediction_label=get_string_labels(prediction)
for input,label in zip(examples,prediction_label):
    print("prediction labels",label.numpy())
