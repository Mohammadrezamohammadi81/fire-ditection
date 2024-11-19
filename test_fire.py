import cv2 
import numpy as np 
from joblib import load
import glob
from keras.models import load_model 

output_label=["fire","non fire"]


for item in glob.glob("test\\*"):
    img=cv2.imread(item)
    imgh=cv2.resize(img,(32,32))
    imgh=imgh/255
    #imgh=imgh.flatten()
    mlp=load_model('cnn.h5')
    imgh=np.array([imgh])
    out=mlp.predict(imgh)[0]
    print(out)
    maxpred=np.argmax(out)
    output=output_label[maxpred]
    if output =="fire":
        cv2.putText(img,"fire !!!!",(10,40),cv2.FONT_HERSHEY_COMPLEX,1.9,(255,255,0),2)

    else:
        cv2.putText(img,"non fire",(10,40),cv2.FONT_HERSHEY_COMPLEX,1.9,(255,255,0),2)    

    
    cv2.imshow("image",img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
    