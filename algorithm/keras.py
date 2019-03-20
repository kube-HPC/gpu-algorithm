from __future__ import absolute_import, division, print_function

import os
import time
import json
import sys

import tensorflow as tf
from tensorflow import keras

def exit(args):
  code = 0
  print("args:", args)
  if (args):
      code = args.get('exitCode', 0)
  print('Got exit command. Exiting with code', code)
  sys.exit(code)

def start(args):
  input=args["input"]
  taskId=args.get("taskId",'stam')
  train_size=input[0].get("train_size",1000)
  output=input[0].get("output",'/training_1')
  num_epochs=input[0].get("num_epochs",10)
  print('started with train_size={train_size}, output={output}, num_epochs={num_epochs}'.format(train_size=train_size,output=output,num_epochs=num_epochs))
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

  train_labels = train_labels[:train_size]
  test_labels = test_labels[:1000]

  train_images = train_images[:train_size].reshape(-1, 28 * 28) / 255.0
  # train_images = train_images.reshape(-1, 28 * 28) / 255.0
  test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
  checkpoint_dir = output+'/'+taskId
  checkpoint_path = checkpoint_dir+"/cp.ckpt"

  # Create checkpoint callback
  cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                  save_weights_only=True,
                                                  verbose=1)
  log_callback = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir+'/logs')

  model = create_model()
  model.summary()
  start = time.time()
  print("Train start")

  model.fit(train_images, train_labels,  epochs = num_epochs, 
            validation_data = (test_images,test_labels),
            callbacks = [cp_callback, log_callback])  # pass callback to training
  end = time.time()
  print("Train time: {}".format(end - start))

  # Recreate the exact same model, including weights and optimizer.
  new_model = create_model()
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  new_model.load_weights(latest)
  loss, acc = new_model.evaluate(test_images, test_labels)
  print("Restored model, accuracy: {:5.2f}%".format(100*acc))

  
  return {'loss':loss.item(), 'acc':acc.item(), 'output': latest}

# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])
  
  model.compile(optimizer='adam', 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model




