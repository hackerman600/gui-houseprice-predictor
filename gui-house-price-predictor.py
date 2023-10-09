import requests
import tkinter as tk
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#fetch the data
data = pd.read_csv("/home/kali/Desktop/projects/out.csv",header=None)
data = np.array(data)
x = data[:,:6]
y = data[:,6:]

print(f"shape of data = {x.shape}")
print(f"shape of data = {y.shape}")


#fix the data
x[:,0] *= 100
x[:,1] /= 20
x[:,1] = np.round(x[:,1])
x[:,2] = np.round(x[:,2] - 1)
x[:,3] = x[:,3] / 4
x[:,3] = np.round(x[:,3])
x[:,5] = np.round(x[:,5] * 5)


#create the columns (did not come with any). print first training example
x_columns = ["land size", "bathrooms", "bedrooms", "garages", "house size", "age"]
print(x[0])

#create and train model ------> nn initialises weights using xaiver glorot by default. this samples from uniform or normal dist with shifted variance and mean of 0, exploding gradients because the loss is so huge to start.  
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(6,)))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dense(1,activation = "linear"))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

#model.summary()
model.fit(x,y,epochs=1000)"""

model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
mse = mean_squared_error(y,y_pred)

n = np.array([5,5,5,5,5,5])
n = n.reshape(1,n.shape[0])
print("n.shape = ", n.shape)


pred = model.predict(n)
print(pred)



def predict_price(x):
    y_pred = model.predict(x)
    return y_pred


def print_price(x,canvas):
    n = []
    for i in x:
        n.append(int(i))

    n = np.array(n)
    n = n.reshape(1,n.shape[0])
    pred = model.predict(n)

    my_pred = tk.Label(canvas,text=f"estimated price: ${int(pred[0])}",background="black",foreground="red",font=("Helvetica",15))
    my_pred.place(x = 250, y = 180)


#create gui 
def application():
    root = tk.Tk()
    root.geometry("500x500")
    root.title("gui-houseprice-predictor.")

    canvas = tk.Canvas(root,background="black",width=501,height=501)
    canvas.place(x = -1, y = -1)

    title_label = tk.Label(canvas, text="houseprice-predictor",foreground="white", background="black",font=("helvetica",20))
    title_label.place(x = 130, y = 60)

    land_entry = tk.Entry(canvas,background="white")
    land_entry.place(x = 130, y = 120, width=60)

    land_label = tk.Label(canvas,text="land size",background="black",foreground="red")
    land_label.place(x = 125, y = 150)

    bathroom_entry = tk.Entry(canvas,background="white")
    bathroom_entry.place(x = 130, y = 178, width=60)

    bathroom_label = tk.Label(canvas,text="#bathrooms",background="black",foreground="red")
    bathroom_label.place(x = 120, y = 208)

    bedroom_entry = tk.Entry(canvas,background="white")
    bedroom_entry.place(x = 130, y = 238, width=60)

    bedroom_label = tk.Label(canvas,text="#bedrooms",background="black",foreground="red")
    bedroom_label.place(x = 120, y = 268)

    garage_entry = tk.Entry(canvas,background="white")
    garage_entry.place(x = 130, y = 298, width=60)

    garage_label = tk.Label(canvas,text="#garages",background="black",foreground="red")
    garage_label.place(x = 125, y = 328)

    housesize_entry = tk.Entry(canvas,background="white")
    housesize_entry.place(x = 130, y = 358, width=60)

    housesize_label = tk.Label(canvas,text="house size",background="black",foreground="red")
    housesize_label.place(x = 125, y = 388)

    age_entry = tk.Entry(canvas,background="white")
    age_entry.place(x = 130, y = 418, width=60)

    age_label = tk.Label(canvas,text="age",background="black",foreground="red")
    age_label.place(x = 145, y = 448)
    #["land size", "bathrooms", "bedrooms", "garages", "house size", "age"]
    my_button = tk.Button(canvas, text="estimate price", background="red", foreground="black",height=2, command= lambda: print_price([land_entry.get(),bathroom_entry.get(),bedroom_entry.get(),garage_entry.get(),housesize_entry.get(),age_entry.get()],canvas))
    my_button.place(x = 250, y = 120)










   

    root.mainloop()


application()
