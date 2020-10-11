#/**
#* Copyright (c) 2020, Vsevolod Averkov <averkov@cs.petrsu.ru>
#*
import cv2 as cv2 
try:
	from keras.models import load_model
except:
	from tensorflow.keras.models import load_model
import pickle

def process(imagepath):
		try:
				image = cv2.imread(str(imagepath))
				print(imagepath)
				output = image.copy()
				image = cv2.resize(image, (224,224))
				image = image.reshape((1, image.shape[0], image.shape[1],
				image.shape[2]))

		except:
			print ("Неверный формат файла либо неверный путь к файлу")
			exit(1)

		print("Загружаю модель")
		try:

			model = load_model(str("imageprocessing\\resouces\\model.h5"))
		
		except:
			print ("Неверный формат файла либо неверный путь к файлу модели")
			exit(1)
	    
		try:
			lb = pickle.loads(open(str("imageprocessing\\resouces\\pic.pickle"), "rb").read())
        
		except:

				print("Неверный формат файла либо неверный путь к файлу меток")
				exit(1)
		
	     
		preds = model.predict(image)


		i = preds.argmax(axis=1)[0]
		label = lb.classes_[i]

	   
   
		if (label==str("VolkswageTiguan")):
		   	label = "VolkswagenTiguan"
		
		return label,preds,output
		   
def draw (label,preds,output,name_of_photo):
		text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
		cv2.putText(output, text, (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
			(0, 200, 0), 2)
		cv2.imwrite(str(name_of_photo)+".jpg", output)
		
