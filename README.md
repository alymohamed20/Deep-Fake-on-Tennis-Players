# Deep-Fake-on-Tennis-Players
This project is used to generate frames or images of fake person swapped with the real professional tennis player. 

## Dataset:
There are two datasets used one of them is downloaded from the Wimbledon championship channel on YouTube as video, then the video is cut in frames and cropped to be ready to use by a python script. The second one is the images of different poses of an unknown person. They are in jpg format and the size of two dataset combined about 6 Gb.

 ![image](https://user-images.githubusercontent.com/73744812/187739123-7a85e423-b28a-420e-a9d8-1ceb1e77cc1b.png)
  ![image](https://user-images.githubusercontent.com/73744812/187739130-d462752f-407f-4ee1-b996-6268936e0231.png)
  ![image](https://user-images.githubusercontent.com/73744812/187739176-3a74cccb-ebab-4a20-9549-b52cd28aa684.png)
  ![image](https://user-images.githubusercontent.com/73744812/187739554-770b0548-7d24-4c05-8d21-7861aeac279b.png)



## Project Methodolgy:
The Cycle GAN is used in our methodolgy to solve the Unpaired image to image translation problem. The cycle consistency loss is implemented to keep the two generators acquired mappings from conflicting with one another. The generator network has 2x2 strides and 9 ResNet blocks for 256x256 pictures by using instance normalization. In the Discriminator they use a Patch GAN 70x70 which is used to determine the 70x70 picture patch is real or fake and it has less parameters than other features.
