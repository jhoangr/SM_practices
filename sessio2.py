#from scipy import misc
import cv2
from scipy.fftpack import dct, idct
from PIL import Image,ImageDraw  # necessari tenir instalar llibreria PILLOW
import numpy as np
import metrikz
import time

quantization_matrix =  [[16.,11.,10.,16.,24.,40.,51.,61.],
                        [12.,12.,14.,19.,26.,58.,60.,55.],
                        [14.,13.,16.,24.,40.,57.,69.,56.],
                        [14.,17.,22.,29.,51.,87.,80.,62.],
                        [18.,22.,37.,56.,68.,109.,103.,77.],
                        [24.,35.,55.,64.,81.,104.,113.,92.],
                        [49.,64.,78.,87.,103.,121.,120.,101.],
                        [72.,92.,95.,98.,112.,100.,103.,99.]]

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantization_process(block):
    qm=np.zeros((8, 8))
    for i in range(int(len(block))):
        for j in range(int(len(block))):
            qm[i][j] = np.round( float(block[i][j]) / quantization_matrix[i][j])
    return qm

# Aplicar DCT y llamar la funcion de cuantizacion
def func_quantized(block):
    qm=np.zeros((8, 8),dtype=int)  
    qm[:,:] = quantization_process(dct2(block[:,:]))
    return qm


# El inverso de la operacion anterior
def func_inverse(block):
    inverse = inverse_process(block)
    inverse = round(idct2(block_struct.data * quantization_matrix));





if __name__ == '__main__':

    #frame anterior
    I1 = Image.open("frame0_1.png")
    #frame actual
    I2 = Image.open("frame0_2.png")

#   I1.show()
    img1 = I1.convert('L')
    img1.save('frame0_gray.png')
    img2 = I2.convert('L')
    img2.save('frame1_gray.png')
#   img1.show()
    #draw = ImageDraw.Draw(I2) 
    #draw.line((4,12, 12,4), fill=(0,53,0),width=1)
    #I2.show()
    #I2.save('')
   
    frame1 = cv2.imread("frame0_gray.png")
    frame2 = cv2.imread("frame1_gray.png")
    #cv2.imshow('image',frame1)
    #img1.show()
    frame1=frame1[:,:,0]
    frame2=frame2[:,:,0]
    print(frame1)
    
  
    print("dimensions de la imatge= "  )
    print(frame1.shape)
    dim=frame1.shape
    
    #vectors finals
    actual_position=[]
    motion_vector=[]
    errors_prediccio=[]
       
    # matriu per generar els blocs 
    bl=np.zeros((8, 8))
    #print(bl)
    #frame1[0:8,0:8]=bl
    bl_2compare=np.zeros((8, 8))
   
  
    #print("frame1",frame1)
    #print("\n")
    #print("\n")
    #print("frame2",frame2)
    #print("\n")
    #print("\n")
    #valor de mse grande
    mse_actual=float("inf")
    mse_mitja=0.0
   # print("mse_actual",mse_actual)
   #
   #
   # GERENAR EL CODI PER FER EL MOTION VECTORS
   #
   # afegir a vector actual_positions totes les coordenades dels blocks de les imatges
   # afegir a vector motion_vector coordenades el block que menor error te del frame anterior
   # afegir a vector errors_prediccio l'error quantitzat !! (func_quantized) que es comet per canviar de block al seguent frame
   #
   #
    start=time.perf_counter()
    for i in range (int(frame2.shape[0]/8)):
        for j in range(int(frame2.shape[1]/8)):
            bl=frame2[i*8:i*8+8,j*8:j*8+8]
            mse_actual=float("inf")
            pos_fin=(i*8,j*8);
            actual_position.append(pos_fin)
            #print("bloque del frame2 i,j:",i,j,"->>>",bl)
            #cerquem en el frame1 el bloc del frame2
            for dx in range (frame1.shape[0]):
                for dy in range (frame1.shape[1]):
                    if((dx+8 <= (frame1.shape[0]) ) and (dy+8<= (frame1.shape[1]))):
                        bl_2compare=frame1[dx:dx+8,dy:dy+8]
                        min_mse=min(metrikz.mse(bl_2compare, bl), mse_actual)
                        if(mse_actual > min_mse):
                           mse_actual=min_mse                    
                           pos_ini=(dx,dy)
            motion_vector.append(pos_ini)
            error_pred=frame2[pos_fin[0]:pos_fin[0]+8,pos_fin[1]:pos_fin[1]+8]-frame1[pos_ini[0]:pos_ini[0]+8,pos_ini[1]:pos_ini[1]+8] 
            #print("error_pred",error_pred)
            error_pred=dct2(error_pred)
            error_pred=quantization_process(error_pred)
            errors_prediccio.append(error_pred)
            mse_mitja=mse_mitja+mse_actual
            #print("mse_final",mse_actual,"... frame2 ij:",i,j)
    print("Temps total:",time.perf_counter()-start)
    print((frame2.shape[0]/8)*frame2.shape[1]/8)
    mse_mitja=mse_mitja/((frame2.shape[0]/8)*frame2.shape[1]/8)       
    print("mse_mitja:",mse_mitja)
    print("\n")
    print("motion_vector",motion_vector)
    print("\n")
    print("actual_position",actual_position)
    print("\n")
   #print("errors_preddicio",errors_prediccio)
   
            
            
            
 

    #
    #
    # GENERAR AQUI EL CODI DE VISUALITZACIO
    #
    #
    draw = ImageDraw.Draw(I2) 
    for i,element in enumerate(motion_vector):
        if element != actual_position[i]:
            
            draw.line((element[1]+4,element[0]+4, actual_position[i][1]+4,actual_position[i][0]+4), fill="red",width=1)
    I2.show()
    I2.save('moviments1_img.png')
  

    
    
