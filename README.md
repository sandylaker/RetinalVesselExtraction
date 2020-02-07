# RetinalVesselExtraction
A project for the DRIVE Challenge.

---
### Description: 
Vessel segmentation from retinoscopic images using UNet or UNet++. The original dataset contains only 21 training images, we augmented this using a randomized deformation.

---
### Note:
When cloning the project in to AISE jupyter notebook VM on Google Cloud, do following:
1. Enter `sudo -i` to enter sudo mode
2. `cd /jet/prs/workspace` to enter the root directory
3. `git clone <Project URL>`
4. After clone the project, we need to set up the writing permission, using `chmod -R 777 <Project Name>`

---
In the function `train` in `src.train`, after `model.to(device)`, we need
to move the optimizer values to GPU as well.
See [Loading a saved model for continue training
](https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4)
