import io
import os
import re
import shutil
import string
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Model, Sequential, layers, losses
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# install the dataset:
# 1)
# 1.1)
# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# to install dataset
# dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')
# after install become like this
# 1.2)
dataset = './aclImdb_v1.tar.gz'
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(os.listdir(dataset_dir))
# train dataset directory:
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# 1.3)
# take simple and open and read file
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# 1.4)
#  remove all files without pos and neg
# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)

# 2)
# 2.1)
# spilt the training data to tow data train , validation
batch_size = 32
seed = 42
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

# 2.2)
#  take some of train data and show the label
# for text_batch, label_batch in raw_train_ds.take(1):
#   for i in range(3):
#     print("Review", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])

# 2.3)
# show what is a label line to class
# print("Label 0 corresponds to", raw_train_ds.class_names[0])
# print("Label 1 corresponds to", raw_train_ds.class_names[1])

# 2.4)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
# 2.5)
# test data
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)


# 3)
# 3.1)
# this function to take the sentences and comes in lowercase then delete the htmal tags then return the sentences
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


# 3.2)
# use TextVectorization to tokenize and standardize and vectoriz the sentences
# textvectorization(standardize:how will standardize,max_tokens:max number to tokens,output_mode:convert the token to  unique int,output_sequence_length:lenth of the output)
max_features = 10000
sequence_length = 250
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# 4)
# train text= raw of train data without labels just review
#  call adapt to add index to train data
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# 4.1)
# this function to know the result of preprocessing data
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# 4.2)
# we take example of train data and see the result

# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

# every wo it get the value,this example of the number --->

# print("1350 ---> ",vectorize_layer.get_vocabulary()[1350])
# print(" 2 ---> ",vectorize_layer.get_vocabulary()[2])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# all data recover to tensor
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# 4.3)
# we use cache() to load the files in memory and build cache in desk if your dataset bigger than the memory
# prefetch() overlaps data preprocessing and model execution  during training
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5)
# create the model:
embedding_dim = 16
# model contain three layers :
# A)the first layer (embedding):
#   take the integer-encoded reviews and looks up an embedding vector for each word-index
#   this layers learned in train,the output is (batch, sequence, embedding)
#   the input shape (2D)(batch_size,input_length) --> the output shape 3D(batch_size,input_length,output_dim)
#   this layer can use just in first layer in model
# B)the Dropout layer:
#   it layers help to prevent the overfitting.
#   the dropout layer is a technique select randomly nods droped-out so all node  learn independent
#   0.2 --> 20% mean 1 in 5 inputs will be randomly droped_out from each update cycle
# C)the second layer is (GlobalAvaragePooling1D):
#   returns a fixed-length output vector for each example by averaging over the sequence dimension.
#   This allows the model to handle input of variable length, in the simplest way possible.
#   input shape (3D) --> output shape (2D)
# D)the third layer is (dense):
#   input shape (2D) --> output shape ( numbers of classes)
#   is a fully connected layers which preform classification on the features extracted and downsampled by the pooling layers
#   the final node we have single node for each target class
#   here we need just single target node
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()

# 6)
# we need lose and optimizer function:
# we have binary classification so we need to loss function : BinaryCrossentropy
# we have BinaryCrossentropy if we have two class
# we have CategoricalCrossentropy if we have more two class. labels provided in a one_hot
# we have SparseCategoricalCrossentropy if we have more two class. labels provided in integer
# optimizer : adam
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# 7)
# train the model

epochs = 10
checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_ds,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback])

# 8)
# test the model:
loss, accuracy = model.evaluate(test_ds)
print("test the model")
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# 9)
# export the model:
# create new model and add the model to the new model
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
#
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)


# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print("test exbort_model")
print(accuracy)
export_model.save_weights('checkpoint1/cp.ckpt')


examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

print(export_model.predict(examples))
print("finish")
