import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt



df=pd.read_csv("Churn_Modelling.csv")


df.drop(["CustomerId","RowNumber","Surname"],axis=1,inplace=True)


df=pd.get_dummies(df,columns=["Geography","Gender"],drop_first=True,dtype=int)

X=df.drop("Exited",axis=1)
Y=df["Exited"]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


model=Sequential([
    Dense(16,activation="relu",input_shape=(X_test_scaled.shape[1],)),
    Dense(16,activation="relu"),
    Dense(1,activation="sigmoid")
])

# model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history= model.fit(X_train_scaled,Y_train,epochs=100,batch_size=16,validation_split=0.2)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.show()

y_pred=model.predict(X_test_scaled)
y_pred_convert=(y_pred>0.5).astype(int)

print("Accuracy:",accuracy_score(Y_test,y_pred_convert))
print("Confusion Matrix:\n",confusion_matrix(Y_test,y_pred_convert))

