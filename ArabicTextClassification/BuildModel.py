import tensorflow as tf
from tensorflow.keras import models, layers, losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from testArabicTextModel.csv_to_txt import csv_to_txt_train_test_group_with_preprocess
from nltk.tokenize import RegexpTokenizer
import nltk
import os
import string
import re

dataset_dir = '/home/ahmad/Desktop/classificationTextProject/testArabicTextModel/archive'
model_dir = '/home/ahmad/Desktop/classificationTextProject/testArabicTextModel/model'

# 1)
# Database spilt classes to train and test with preprocess
#############################################################################
for file in os.listdir(dataset_dir):
    if file.endswith('.csv'):
        file_dir = os.path.basename(file)
        class_name = os.path.splitext(file_dir)[0]
        csv_file_dir = "{}/{}".format(dataset_dir, file)
        csv_to_txt_train_test_group_with_preprocess(csv_file_dir, class_name, ['story'])
#############################################################################

# 2)
# Preperations the train,validate,test Dataset

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
#############################################################################
# print(train_dir)
# print(test_dir)
#############################################################################
batch_size = 32
seed = 42
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, batch_size=batch_size, seed=seed,
                                                                  subset='training', validation_split=0.2)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, batch_size=batch_size, seed=seed,
                                                                subset='validation', validation_split=0.2)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir, batch_size=batch_size)
#############################################################################
# for text_batch, label_batch in raw_train_ds.take(3):
#     for i in range(5):
#         print("Review", text_batch.numpy()[i])
#         print("Label", label_batch.numpy()[i])
# print("Label 0 corresponds to", raw_train_ds.class_names[0])
# print("Label 1 corresponds to", raw_train_ds.class_names[1])
# print("Label 2 corresponds to", raw_train_ds.class_names[2])
# print("Label 3 corresponds to", raw_train_ds.class_names[3])
# print("Label 4 corresponds to", raw_train_ds.class_names[4])
# print("Label 5 corresponds to", raw_train_ds.class_names[5])
# print("Label 6 corresponds to", raw_train_ds.class_names[6])
# print("Label 7 corresponds to", raw_train_ds.class_names[7])
# print("Label 8 corresponds to", raw_train_ds.class_names[8])
# print("Label 9 corresponds to", raw_train_ds.class_names[9])
# print("Label 10 corresponds to", raw_train_ds.class_names[10])
#############################################################################


max_features = 200000
sequence_length = 1000
vectorize_layer = TextVectorization(max_tokens=max_features,
                                    output_mode='int',
                                    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


#############################################################################
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))
# print("1350 ---> ",vectorize_layer.get_vocabulary()[1350])
# print(" 2 ---> ",vectorize_layer.get_vocabulary()[2])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
#############################################################################

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
auto_tune = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=auto_tune)
val_ds = val_ds.cache().prefetch(buffer_size=auto_tune)
test_ds = test_ds.cache().prefetch(buffer_size=auto_tune)

# 3)
# build the model:

embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.5),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    layers.Dense(11, activation='softmax')])

model.summary()
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.SparseCategoricalAccuracy())
epochs = 128
checkpoint_path = os.path.join(model_dir, "checkpointmodel/cp.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit(train_ds,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback])
loss, accuracy = model.evaluate(test_ds)
print("test the model")
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# 4)
# build export model:

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print("test export_model")
print(accuracy)
export_model.save_weights('checkpointExportModel/cp.ckpt')

#
# examples = ["how do i extract keys from a dict into a list?",
#             "debug public static void main(string[] args){...}"
# ]
#
# def get_string_labels(batch):
#     predicted_int_labels= tf.argmax(batch,axis=1)
#     predicted_labels=tf.gather(raw_train_ds.class_names,predicted_int_labels)
#     return predicted_labels
#
# prediction = export_model.predict(examples)
# prediction_label=get_string_labels(prediction)
# for input,label in zip(examples,prediction_label):
#     print("prediction labels",label.numpy())
#
# print(export_model.predict(examples))
print("finish")
