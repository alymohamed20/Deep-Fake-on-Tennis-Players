# Deep-Fake-on-Tennis-Players
This project is used to generate frames or images of fake person swapped with the real professional tennis player. 

## Project methodolgy:
The Cycle GAN is used in our methodolgy to solve the Unpaired image to image translation problem. The cycle consistency loss is implemented to keep the two generators acquired mappings from conflicting with one another. The generator network has 2x2 strides and 9 ResNet blocks for 256x256 pictures by using instance normalization. In the Discriminator they use a Patch GAN 70x70 which is used to determine the 70x70 picture patch is real or fake and it has less parameters than other features.
