# Deep-Fake-on-Tennis-Players
This project is used to generate frames or images of fake person swapped with the real professional tennis player. 

## Dataset:
There are two datasets used in our project one of them is downloaded from the Wimbledon championship channel on YouTube as video, then the video is cut in frames and cropped to be ready to use by a python script. The second one is the images of different poses of an unknown person. They are in jpg format and the size of two dataset combined about 6 Gb.

 ![image](https://user-images.githubusercontent.com/73744812/187739123-7a85e423-b28a-420e-a9d8-1ceb1e77cc1b.png)
  ![image](https://user-images.githubusercontent.com/73744812/187739130-d462752f-407f-4ee1-b996-6268936e0231.png)
  ![image](https://user-images.githubusercontent.com/73744812/187739729-127b5e0a-d45d-426d-8d3d-f413ffb623b9.png)
  ![image](https://user-images.githubusercontent.com/73744812/187739753-462841ae-8b76-4b6c-8fb6-711df26a08f8.png)
  
## Libraries used in this Project:
•	Python version 3.7.8

•	TensorFlow version 2.6.0

•	Keras version 2.6.0

•	Matplotlib version 3.3.2

•	NumPy version 1.19.2





## Project Methodolgy:
The Cycle GAN is used in our methodolgy to solve the Unpaired image to image translation problem. The cycle consistency loss is implemented to keep the two generators acquired mappings from conflicting with one another. The generator network has 2x2 strides and 9 ResNet blocks for 256x256 pictures by using instance normalization. In the Discriminator we use a Patch GAN 70x70 which is used to determine the 70x70 picture patch is real or fake and it has less parameters than other features.

![Screenshot 2022-08-31 191914](https://user-images.githubusercontent.com/73744812/187740282-b5d4f94b-4b74-41aa-a03a-78f8450238c2.jpg)

## Acceptanle Results:
![BtoA_generated_plot_007760](https://user-images.githubusercontent.com/73744812/187742299-7d2448c3-176e-4af9-95ff-92e8ace94185.png)
![BtoA_generated_plot_004675](https://user-images.githubusercontent.com/73744812/187742314-06a5e937-5b6c-4be2-9661-cf9b59ffeef8.png)
![image](https://user-images.githubusercontent.com/73744812/187742454-8b5b1071-12fe-49db-8f14-848b8a8ee0e7.png)
![image](https://user-images.githubusercontent.com/73744812/187742537-77d820d9-91dc-42a6-affe-47bb4b4796fb.png)



