import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tkinter import Label
from tkinter import Button
import numpy
#load the trained model to classify sign
from tensorflow.keras.models import load_model
model = load_model('prueba1.h5')
#dictionary to label all traffic signs class.
classes = { 1:'Avion',
            2:'Automovil', 
            3:'Pajaro', 
            4:'Gato', 
            5:'Venado', 
            6:'Perro', 
            7:'Sapo', 
            8:'Caballo', 
            9:'Barco', 
            10:'Camión' }
#initialise GUI
top=tk.Tk()
top.geometry('800x800')
top.title('CIFAR-10 CLASSIFIER')
top.configure(background='white')
label=Label(top,background='white', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
def show_classify_button(file_path):
    classify_b=Button(top,text="Clasificar imagen",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Cargar una imagen",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Conoce cuál es la Imagen",pady=20, font=('arial',20,'bold'))
heading.configure(background='white',foreground='#364156')
Clase1= Label(text="Estas Son las Clases",pady=20, font=('arial',10,'bold'))
heading.pack()
top.mainloop()
