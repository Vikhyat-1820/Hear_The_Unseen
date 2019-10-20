
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model


vgg_model=VGG16()
vgg_model.layers.pop()
vgg_model=Model(inputs=vgg_model.inputs,outputs=vgg_model.layers[-1].output)

def feature_extractor(imgg):
        img=image.img_to_array(imgg)
        img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        img=preprocess_input(img)
        features=vgg_model.predict(img)
        return features