import cv2
import numpy as np

width = 1280
height = 720

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3, width)
cap.set(4, height)

window_name = "image"

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



class Vec:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy



paddleX = width - 230

ball = Vec(100, 100, 10, 10)
ball.x = int(width/2)
ball.y = int(height/2)
ball.dy = 20
ball.dx = -20

leftScore = 0
rightScore = 0



while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)

    ball.x += ball.dx
    ball.y += ball.dy

    if ball.y > height - 5:
        ball.y = height - 5
        ball.dy *= -1

    if ball.y < 0:
        ball.y = 0
        ball.dy *= -1

    if ball.x > width - 5:
        ball.x = int(width / 2)
        ball.y = int(height / 2)
        ball.dy = 20
        ball.dx = -20
        # add score for left player here
        leftScore = leftScore + 1

    if ball.x < 0:
        ball.x = int(width / 2)
        ball.y = int(height / 2)
        ball.dy = 20
        ball.dx = -20
        # add score for right player here
        rightScore = rightScore + 1

    faceCords = []

    for (x, y, w, h) in faces:
        faceCords.append(Vec(x, y, w, h))

    faceCords.sort(key=lambda c: c.x)

    for index, vec in enumerate(faceCords[0:2], start=0):
        cv2.rectangle(img, (100 + (index * paddleX), vec.y), (100 + (index * paddleX) + 30, vec.y + 100), (100, 100, 0), 1)
        delX = abs(ball.x - (100 + (index * paddleX) + (30 * (1 - index)))) * (ball.dx / 10)

        if ball.x == (100 + (index * paddleX) + ((1 - index) * 30)) and vec.y <= ball.y <= vec.y + 100:
            print("bounce: ", ball.dx)
            ball.dx *= -1
        elif (100 + (index * paddleX)) <= ball.x + ball.dx <= (
                100 + (index * paddleX) + ((1 - index) * 30)) and vec.y <= ball.y + ball.dy <= vec.y + 100:
            ball.y = int(delX + ball.y)
            ball.x = 100 +(index * paddleX) + ((1 - index) * 30) - ball.dx
            print(ball.dx, ball.x)

    cv2.circle(img, (ball.x, ball.y), 9, (0, 0, 255), -1)
    text1 = cv2.putText(img, 'Left Player Score: ' + str(leftScore), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    text2 = cv2.putText(img, 'Right Player Score: ' + str(rightScore), (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
