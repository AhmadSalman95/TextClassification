import tensorflow as tf
from tensorflow.keras import models,layers,losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import os
import string
import re


dataset='stack_overflow_16k.tar.gz'
dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')
# print(os.listdir(dataset_dir))
train_dir=os.path.join(dataset_dir, 'train')
# print(os.listdir(train_dir))

batch_size=32
seed=42
raw_train_ds=tf.keras.preprocessing.text_dataset_from_directory('stack_overflow_16k/train',batch_size=batch_size,seed=seed,subset='training',validation_split=0.2)
raw_val_ds=tf.keras.preprocessing.text_dataset_from_directory('stack_overflow_16k/train',batch_size=batch_size,seed=seed,subset='validation',validation_split=0.2)
raw_test_ds=tf.keras.preprocessing.text_dataset_from_directory('stack_overflow_16k/test',batch_size=batch_size)
# for text_batch, label_batch in raw_train_ds.take(3):
#   for i in range(5):
#     print("Review", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])
# print("Label 0 corresponds to", raw_train_ds.class_names[0])
# print("Label 1 corresponds to", raw_train_ds.class_names[1])
# print("Label 2 corresponds to", raw_train_ds.class_names[2])
# print("Label 3 corresponds to", raw_train_ds.class_names[3])

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
max_features=100000
sequence_length=1000
vectorize_layer=TextVectorization(max_tokens=max_features,standardize=custom_standardization,output_mode='int',output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))
# print("1350 ---> ",vectorize_layer.get_vocabulary()[1350])
# print(" 2 ---> ",vectorize_layer.get_vocabulary()[2])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.5),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')])

model.summary()
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.SparseCategoricalAccuracy())
epochs = 128
checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
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

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
#
export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print("test exbort_model")
print(accuracy)
export_model.save_weights('checkpointExportModel/cp.ckpt')

examples = ["how do i extract keys from a dict into a list?",
            "debug public static void main(string[] args){...}"
]

def get_string_labels(batch):
    predicted_int_labels= tf.argmax(batch,axis=1)
    predicted_labels=tf.gather(raw_train_ds.class_names,predicted_int_labels)
    return predicted_labels

prediction = export_model.predict(examples)
prediction_label=get_string_labels(prediction)
for input,label in zip(examples,prediction_label):
    print("prediction labels",label.numpy())

# print(export_model.predict(examples))
print("finish")