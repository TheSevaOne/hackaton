#/**
#* Copyright (c) 2020, Vsevolod Averkov <averkov@cs.petrsu.ru>
#*
import matplotlib
matplotlib.use("Agg")
import numpy as np
from keras.models import Sequential , Model
from keras.layers import Dropout, Dense, Flatten,Activation,AvgPool2D
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.applications import VGG16, ResNet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD,Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.models import load_model
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="img")
ap.add_argument("-m", "--model", required=True,
	help="model")
ap.add_argument("-l", "--label-bin", required=True,
	help="label ")
ap.add_argument("-p", "--plot", required=True,
	help="picture")
args = vars(ap.parse_args())

print("[INFO] Загружаю картинки")
data = []
labels = []

# захватить пути изображения и случайным образом перемешать их
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(200)
random.shuffle(imagePaths)
i=0
print(imagePaths)
for imagePath in imagePaths:
	try:
       
		print(imagePath)
	
		
		image = cv2.imread(imagePath)

		image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_CUBIC)
		data.append(image)
       
		# извлечь метку класса из пути к изображению и обновить
		# список меток
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
    
	except:

		print(len(labels))

		
data = np.array(data, dtype="float")
labels = np.array(labels)


# разбить данные на разделы обучения и тестирования, используя 75%
# данные для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=200)

# конвертировать метки из целых чисел в векторы 
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

from keras.models import Sequential

def build(tensor_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape= tensor_shape)
    
    model = base_model.output
    
    model = AvgPool2D()(model)
    
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.5)(model)
    
    predictions=(Dense(len(lb.classes_), activation="softmax"))(model)
    
    model_ok = Model(inputs=base_model.input, outputs=predictions)
    
    return model_ok

tensor_shape = (224,224,3)

model = build(tensor_shape)
#model.load_weights("sec.h5")
INIT_LR = 0.0001
EPOCHS = 120
print("[INFO] тренерую")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# тренеровка


H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32,verbose=2)
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


score = model.evaluate(testX, testY, verbose=0)
print("Точность: %.2f%%" % (score[1]*100))

# графики
try:
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.title("Тренировка нейросети  Loss and Accuracy")
	plt.xlabel("Эпохи")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(args["plot"])
except:
	pass

print("[INFO] сохраняю")
#model.save_weights("for_t_weights.h5")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
