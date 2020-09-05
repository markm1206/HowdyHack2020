import gpiozero as gpio
from time import sleep
import RPi.GPIO as IO


#servo = gpio.AngularServo(18,0,0,270)
servo = gpio.Servo(18,initial_value = 0, min_pulse_width = .0005, max_pulse_width = 0.00245, frame_width = .02)
#IO.setmode(IO.BCM)
#IO.setup(18,IO.OUT)
delay = 1
angle = 0

servo.detach()

while True:
    servo.min()
    sleep(delay)
    servo.mid()
    sleep(delay)
    servo.max()
    sleep(delay)
    servo.mid()
    sleep(delay)


'''
while True:
    if servo.angle == 270.0:
        angle = 0
    servo.angle = angle 
    print("pos:",angle)
    sleep(delay)
    angle += 5 '''