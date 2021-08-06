#from scipy import misc
import cv2
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import metrikz
import eines_sessio3 as s3
import sys

quantization_matrix =  [[16.,11.,10.,16.,24.,40.,51.,61.],
                        [12.,12.,14.,19.,26.,58.,60.,55.],
                        [14.,13.,16.,24.,40.,57.,69.,56.],
                        [14.,17.,22.,29.,51.,87.,80.,62.],
                        [18.,22.,37.,56.,68.,109.,103.,77.],
                        [24.,35.,55.,64.,81.,104.,113.,92.],
                        [49.,64.,78.,87.,103.,121.,120.,101.],
                        [72.,92.,95.,98.,112.,100.,103.,99.]]



def func_motion_compensation(actual_position, motion_vector, errors_predicio):
    x=0
    mse_3_2=0.0
    mse_4_2=0.0
    for i in range (int(frame1.shape[0]/8)):
        for j in range(int(frame1.shape[1]/8)):
            posx=motion_vector[x][0]
            posy=motion_vector[x][1]
            #obtenim el frame3 a partir del motion vector
            frame3[i*8:i*8+8,j*8:j*8+8]=frame1[posx:posx+8,posy:posy+8]
            mse_3_2=mse_3_2+metrikz.mse(frame3[i*8:i*8+8,j*8:j*8+8], frame2[i*8:i*8+8,j*8:j*8+8])
            #desfem el procés de quantització i dct aplicat als errors de preddicio
            inverse = np.round(s3.idct2(errors_predicio[x] * quantization_matrix))
            #calculem el frame4 a partir del frame3 i els errors de predicció
            frame4[i*8:i*8+8,j*8:j*8+8]=(frame3[i*8:i*8+8,j*8:j*8+8] + inverse)
            mse_4_2=mse_4_2+metrikz.mse(frame4[i*8:i*8+8,j*8:j*8+8], frame2[i*8:i*8+8,j*8:j*8+8])
            x=x+1
    print("mse_3_2:",mse_3_2/x)
    print("mse_4_2",mse_4_2/x)
    return 0


    



if __name__ == '__main__':

    #frame anterior
    I1 = Image.open("frame2_1.png")
    #frame actual
    I2 = Image.open("frame2_2.png")

    N=8

    #I1.show()
    #I2.show()
    img = I1.convert('L')
    img.save('frame11_gray.png')
    img = I2.convert('L')
    img.save('frame12_gray.png')
    #img.show()

    #inicialitzacio de les matrius de les imatges
    frame1 = np.array(cv2.imread("frame11_gray.png"),dtype=np.int16)
    frame2 = np.array(cv2.imread("frame12_gray.png"),dtype=np.int16)
    frame1=frame1[:,:,0]
    frame2=frame2[:,:,0]
    frame3 = np.zeros(frame1.shape,dtype=np.int16)
    frame4 = np.zeros(frame1.shape,dtype=np.int16)
    
    print("dimensions de la imatge.")
    print(frame1.shape)
    

            


    # crida a funcion metode block matching
    [actual_position, motion_vector, errors_prediction]=s3.func_block_matching(frame1,frame2,N)
    #[actual_position, motion_vector, errors_prediction]=s3.func_block_matchingv3(frame1, frame2)
    # crida a funcio metode motion compensation
    func_motion_compensation(actual_position, motion_vector, errors_prediction)

    

    #tamanys en Bytes dels errors de prediccio abans de run-length encoding (RLE)
    # B tamany que ocupa en bytes RLE
    b=0
    # C tamany que ocupa en bytes tots els errors de predicció junyts sense comprimir.
    c=0
    aux = []
    for BL in errors_prediction:
        aux.extend(s3.func_encoded_values(s3.zigzag(BL)))
        c += sys.getsizeof(BL)
    b = sys.getsizeof(aux)


    print ("\nPas N=",N)
    print ("RLE:",float(b)/1024,"KB")
    print ("errors complet",float(c)/1024,"KB")
    print ("ssim frame3,frame2:", metrikz.ssim(frame3,frame2))
    print ("ssim frame4,frame2:", metrikz.ssim(frame4,frame2))


    ## Per mostrar per pantalla les quatre imatges.
    plt.figure(1,figsize=(16,10))
    plt.rcParams['image.cmap'] = 'gray'
    plt.subplot(221)
    plt.title("I1-frame anterior")
    plt.imshow(frame1,vmin=0,vmax=255)
    plt.subplot(222)
    plt.title("I2-frame actual")
    plt.imshow(frame2,vmin=0,vmax=255)
    plt.subplot(223)
    plt.title("I3-frame motion compensation")
    plt.imshow(frame3,vmin=0,vmax=255)
    plt.subplot(224)
    plt.title("I4-frame I3 + errors prediction")
    plt.imshow(frame4,vmin=0,vmax=255)

## Per mostrar el mapa de calor de les variacions, blau menys error, vermell maxim error
    plt.figure(2,figsize=(14,14))
    plt.rcParams['image.cmap'] = 'jet'
    plt.subplot(211)
    plt.title("I3 - I2")
    vm=np.max(np.absolute(frame3-frame2))
    plt.imshow(np.absolute(frame3-frame2),vmin=0,vmax=vm)#np.max(np.absolute(frame3-frame2)))
    plt.subplot(212)
    plt.title("I4 - I2")
    plt.imshow(np.absolute(frame4-frame2),vmin=0,vmax=vm)#np.max(np.absolute(frame3-frame2)))

    plt.show()



   
 
    
