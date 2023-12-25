from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.text import Label as CoreLabel
import numpy as np
from lane_detect.video_lane_detect import process_frame

from kivy.graphics import Color, Rectangle, Line, InstructionGroup
import cv2
import datetime

class Interface(BoxLayout):
    def __init__(self, **kwargs):
        self.lane_cooordinates = [[0, 0], [0, 480], [640, 0], [640, 480]]
        super().__init__(**kwargs)
        self.image_widget = Image()
        self.add_widget(self.image_widget)
        self.overlay = InstructionGroup()
        self.image_widget.canvas.after.add(self.overlay)

        self.capture = cv2.VideoCapture('example.mp4')
        Clock.schedule_interval(self.update, 1/30.)

    def update(self, *args):
        ret, frame = self.capture.read()
        if ret:
            frame = self.lane_detection(frame)
            frame = self.object_detection(frame)
            buf1 = cv2.flip(frame, 0).tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture1
    
    def lane_detection(self, frame):
        line1, line2 = process_frame(frame)
        if line1 is not None:
            self.lane_cooordinates[3] = [line1[0], line1[1]]
            self.lane_cooordinates[2] = [line1[2], line1[3]]
        if line2 is not None:
            self.lane_cooordinates[0] = [line2[0], line2[1]]
            self.lane_cooordinates[1] = [line2[2], line2[3]]
        print(self.lane_cooordinates)
        lane_coordinates = np.array(self.lane_cooordinates, np.int32)
        lane_coordinates = lane_coordinates.reshape((-1, 1, 2)).astype(np.int32)
        frame = self.draw_lane(frame, lane_coordinates)
        return frame

    def object_detection(self, frame):
        return frame

    def draw_text(self, text, x, y):
        with self.image_widget.canvas.after:
            Color(1, 1, 1, 1)  

            core_label = CoreLabel(text=text, font_size=14)

            core_label.refresh()

            texture = core_label.texture
            size = texture.size
            pos = (x, y - size[1])  

            Rectangle(size=size, pos=pos, texture=texture)

    def draw_lane(self, frame, vertices):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [vertices], color=(255, 0, 0)) 
        alpha = 0.3  
        image = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return image

class MyApp(App):
    def build(self):
        return Interface()

if __name__ == '__main__':
    MyApp().run()
