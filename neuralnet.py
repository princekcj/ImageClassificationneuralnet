
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  
import os
for dirname, _, filenames in os.walk('/kaggle/input/pokemonclassification'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import tensorflow
import matplotlib.pyplot as plt
import matplotlib.image as mig
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

#define all target names for classifaction
root_dir = ('/kaggle/input/pokemonclassification/PokemonData')
classes = os.listdir(root_dir)

#Present array of image data within dataset
c1_path = os.path.join(root_dir, classes[1])
c1_data_path = [os.path.join(c1_path, img) for img in os.listdir(c1_path)]
len(c1_data_path)

for i in range(0, 5):
    img = mig.imread(c1_data_path[i])

    print(i, img.shape)
    
  

IDG = ImageDataGenerator(validation_split=0.25 ,rescale = 1./255, rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')

train_generator = IDG.flow_from_directory(root_dir,target_size=(128 ,128),batch_size=6820 ,class_mode='categorical')
val = IDG.flow_from_directory(root_dir,target_size=(128,128),subset='validation',batch_size=8, class_mode='categorical')

#Split the data into a training set and a test set
((sample_x),(sample_y)) = next(train_generator)
x_train, x_test, y_train, y_test = train_test_split(sample_x,sample_y,test_size=0.30, random_state=1)
train_generator.reset()
train_generator = IDG.flow_from_directory(root_dir,target_size=(128 ,128),batch_size=8 ,class_mode='categorical')

#Present the images in the input form
X, Y = next(train_generator)
for x,y in zip( X,Y ):  
  plt.imshow(x)
  plt.xlabel(classes[y.argmax()])
  plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
#training model
img_shape=(128,128,3)


model = keras.Sequential(name='Pokemon_cls_5L_Net')
model.add(keras.layers.Conv2D(64,3,input_shape=(img_shape),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(32,3,strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(32,3,strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(len(classes),activation='softmax'))


model.summary()


model.compile(optimizer='Nadam',
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['acc']
             )
 
hist = model.fit (train_generator ,validation_data=val, epochs=50)


plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['acc'],label='accuracy',color='green')
plt.plot(hist.history['val_acc'], label='validation', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))
plt.show()

plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['loss'],label='loss',color='green')
plt.plot(hist.history['val_loss'], label='validation', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yticks(np.arange(0, 8, step= 0.8))
plt.show()


# # classpokemon = train_generator.class_indices
# classpokemon


classpokemon = train_generator.class_indices


print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=8)
print("test loss, test acc:", results)

print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

y_pred = model.predict(x_test)
confusion_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(confusion_matrix)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model')


import zipfile
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


zipf = zipfile.ZipFile('model.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('./model', zipf)
zipf.close()
