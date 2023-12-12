# deep-biomedicine-project

In this repository we have the code to train deeplearning models for microscopy denoising. More specifically, to improve the quality of maximum intensity projection (MIP) images by learning on extended depth of field (EDOF) images. We propose four methods:
1. Unet
2. Multi wavelet CNN (mwcnn)
3. Diffusion based model
4. GAN
The code to train each of these can be found in their respective directories. To train the Unet or the diffusion model, run `unet/unet.py` and `diffusion/diffusion.py` respectively. For the mwcnn and the GAN, run the notebooks in those directories. After training is finished, the models will be stored in a subdirectory called models in the respective subdirectory (e.g. `gan/models/` for gan models)

The data used needs to be stored in data and split into a training and test directory. The test directory will be used for final evaluation in `main.ipynb`, while the training data will be split into training and validation during the training process.

The `src` has code which is used for each model, such as a Imagedataset class used for data loading and a Trainerclass used for training.

Finally, `main.ipynb` shows the final results and compares the performance of each of the models.

The results directory contains examples of the results.

![alt text](https://github.com/AleHD/deep-biomedicine-project/blob/main/Summary.png)

![alt text](https://github.com/AleHD/deep-biomedicine-project/blob/main/GAN_summary.png)
