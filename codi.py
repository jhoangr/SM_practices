import metrikz
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


mse_y=np.zeros([8,11])
ratioc_y=np.zeros([8,11])
quality_x=np.array([1,10,20,30,40,50,60,70,80,90,100])
c=0
for q in range (0,110,10):
    for i in range (1,9):
        I = Image.open("image"+str(i)+".png")
        
        #sin esta línea la imagen 7 falla 
        if I.mode in ("RGBA", "P"): I = I.convert("RGB") 
        I.save("image"+str(i)+".jpg",quality = q) 
        
        
    	
        source = cv2.imread("image"+str(i)+".png")
        ratioc_y[i-1][c]=(os.stat("image"+str(i)+".png").st_size)/(os.stat("image"+str(i)+".jpg").st_size)
        target =cv2.imread("image"+str(i)+".jpg")

        mse_y[i-1][c]=(metrikz.mse(source,target))
    c+=1
        
        


plt.figure(1)
plt.ylabel("MSE")
plt.xlabel('quality')
plt.plot(quality_x,mse_y[0],color="blue",label="image1",linewidth="3.0")
plt.plot(quality_x,mse_y[1],color="orange",label="image2",linewidth="3.0")
plt.plot(quality_x,mse_y[2],color="green",label="image3",linewidth="3.0")
plt.plot(quality_x,mse_y[3],color="red",label="image4",linewidth="3.0")
plt.plot(quality_x,mse_y[4],color="purple",label="image5",linewidth="3.0")
plt.plot(quality_x,mse_y[5],color="brown",label="image6",linewidth="3.0")
plt.plot(quality_x,mse_y[6],color="olive",label="image7",linewidth="3.0")
plt.plot(quality_x,mse_y[7],color="gray",label="image8",linewidth="3.0")
plt.legend(loc="upper right")
plt.title("Variació del MSE en funció del paràmetre quality")

plt.show()

plt.figure(2)

plt.ylabel("Ratio de compressió")
plt.xlabel("quality")
plt.plot(quality_x,ratioc_y[0],color="blue",label="image1",linewidth="3.0")
plt.plot(quality_x,ratioc_y[1],color="orange",label="image2",linewidth="3.0")
plt.plot(quality_x,ratioc_y[2],color="green",label="image3",linewidth="3.0")
plt.plot(quality_x,ratioc_y[3],color="red",label="image4",linewidth="3.0")
plt.plot(quality_x,ratioc_y[4],color="purple",label="image5",linewidth="3.0")
plt.plot(quality_x,ratioc_y[5],color="brown",label="image6",linewidth="3.0")
plt.plot(quality_x,ratioc_y[6],color="olive",label="image7",linewidth="3.0")
plt.plot(quality_x,ratioc_y[7],color="gray",label="image8",linewidth="3.0")
plt.legend(loc="upper right")
        
plt.show()