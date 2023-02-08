import torch
from torch import nn 
from torch.optim import Adam
from PIL import Image
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = torch.load('./obj_detection_model_3.pt', map_location=torch.device('cpu'))
#print(model)

model.eval()

#print(model)


#image_path = './images/ped/pedestrian-friendly-streets_0.jpg'
#image_path = './images/ped/000025.jpg'
image_path = "./images/ped/PennPed00089.png"
video_path = "./images/ped/22.mp4"

fourcc = cv2.VideoWriter_fourcc(*'XVID')

vid = cv2.VideoCapture(video_path)

out = cv2.VideoWriter('output1.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 10 , (int(vid.get(3)),int(vid.get(4))))
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        raise ValueError("No image!")
    frame_size = frame.shape[:2]
    
    img = torchvision.transforms.ToTensor()(image)
    
    with torch.no_grad():
        prediction = model([img.to('cpu')])

    #print(prediction)
    #bbox = prediction[0]["boxes"][0][0]
    #print(int(prediction[0]["boxes"][0][0]))

    image = np.array(torchvision.transforms.ToPILImage()(img))
    #plt.imshow(image)
    for i in range(len(prediction[0]["boxes"])):
        edited = cv2.rectangle(image, (int(prediction[0]["boxes"][i][0]), int(prediction[0]["boxes"][i][1])),
                            (int(prediction[0]["boxes"][i][2]), int(prediction[0]["boxes"][i][3])), (0, 255, 0), 2)
    cv2.imshow('img', image)

    out.write(edited)
    #cv2.waitKey(0)
    #plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vid.release()

out.release()

cv2.destroyAllWindows()
"""
image = Image.open(image_path)
img = torchvision.transforms.ToTensor()(image)
print(img.size())
# load the train data
train_dataset = torchvision.datasets.ImageFolder(root='./images', 
                                                 transform= torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1)
#image = Image.open(image_path)
with torch.no_grad():
    prediction = model([img.to('cpu')])

image = np.array(torchvision.transforms.ToPILImage()(img))
    #plt.imshow(image)
for i in range(len(prediction[0]["boxes"])):
    cv2.rectangle(image, (int(prediction[0]["boxes"][i][0]), int(prediction[0]["boxes"][i][1])),
                        (int(prediction[0]["boxes"][i][2]), int(prediction[0]["boxes"][i][3])), (0, 255, 0), 2)
cv2.imshow('img', image)
cv2.waitKey(0)
"""
