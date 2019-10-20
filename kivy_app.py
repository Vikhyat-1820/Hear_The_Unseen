__author__ = 'vikhyat'
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from image_captioner import image_captioner
import cv2
from keras.preprocessing import image
import pyttsx3
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()


class CamApp(App):

    def build(self):
        print('its start')
        self.captioner=image_captioner()
        print('wow model created')
        self.capture = cv2.VideoCapture(0)
        print('camera starts')
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        img=cv2.resize(frame,(224,224))
        img=image.array_to_img(img)
        text=self.captioner.generate_desc(img)
        speak(text)
        print(text)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()


