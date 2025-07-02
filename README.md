# PET-Reconstruction-Ennhacement

This repository contains a project focused on enhancing **Positron Emission Tomography (PET)** image reconstruction using Python-based techniques, including curve fitting and image interpolation. The goal is to reduce background noise and improve image resolution in reconstructed PET scans.

## Introduction

Positron Emission Tomography (PET) is a nuclear medicine imaging technique used to map metabolic activity in the body. It works by administering biologically active molecules labeled with positron-emitting radionuclides. When these positrons encounter electrons, they annihilate and produce two gamma photons traveling in opposite directions.

By placing detectors opposite each other, the resulting gamma rays can be detected. Measuring the difference in arrival times between many such pairs across multiple angles allows for the creation of radiation flux projections. These are combined using an **Inverse Radon Transformation** to reconstruct an image of the tracer distribution within the body.

This project builds on an understanding of the PET reconstruction process to enhance image quality and reduce noise. After evaluating the limitations of the imaging equipment, we apply a two-peak Cauchy distribution to fit each projection curve:

$$
f(x) = \frac{a}{\pi  \gamma (1+(\frac{x - x_{01}}{ \gamma })^2)} + \frac{a}{\pi  \gamma (1+(\frac{x - x_{02}}{ \gamma})^2)} + y_{\text{off}}
$$

We then analyze how these fitted parameters change with angle, enabling interpolation across angles. This allows us to simulate projections at finer angular resolutions and reconstruct a smoother, denoised image using the Inverse Radon Transform.

## Data Collection

 To gather projection data for image reconstruction, we used two radiation detectors placed opposite each other and electronically linked so that they would only register an event if both detectors detected radiation simultaneously. This method, known as coincidence detection, ensures that only gamma rays traveling directly between the detectors are counted, reducing noise and improving spatial accuracy.

To improve directionality, we manually arranged square metal plates in front of each detector to form a narrow slit. These slits acted as a simple physical filter, allowing radiation to pass only through a well-defined path and enhancing the consistency of measurements.

The radioactive objects were placed on a movable rail system positioned between the detectors. This setup allowed us to move the object incrementally, pausing at specific distances so the detectors could record radiation intensity. After completing one pass, the object was rotated to a new angle, and the process was repeated. This angular scanning generated a full set of projections, which we later used to reconstruct the internal structure of the object using Inverse Radon Transformation.

## Structure and Image Reconstruction Process


### 1. Converting Projection Data into a Sinogram

Data collected from multiple angles was first arranged into a sinogram, which plots intensity (coincidence counts) as a function of position across various angles. This sinogram is the basis for reconstruction using the Inverse Radon Transform, which generates an initial image representing the distribution of radioactivity in the scanned object.

### 2. Curve Fitting with a Two-Peak Cauchy Distribution
To reduce noise and enhance resolution, each 1D projection was curve-fitted using a two-peak Cauchy distribution. This function was selected because:

- The shape naturally models peak broadening and heavy tails observed in real PET data.

- It handles overlapping signal peaks better than Gaussian or Poisson distributions.

- It smooths random statistical noise by approximating the signal with a continuous analytical function.

This fitting improves image quality when the projections are reassembled into a new sinogram.

$$
f(x) = \frac{a}{\pi  \gamma (1+(\frac{x - x_{01}}{ \gamma })^2)} + \frac{a}{\pi  \gamma (1+(\frac{x - x_{02}}{ \gamma})^2)} + y_{\text{off}}
$$
![alt text](https://github.com/CamelSal/PET_Reconstruction-Enhacement/blob/master/figures/original.png?raw=true)
### 3. Parameter Analysis and Interpolation

Each projection’s fit returned a set of parameters (peak positions, width, amplitude, and offset). These parameters were analyzed as a function of projection angle:

- Peak positions followed a sinusoidal pattern and were fitted using trigonometric functions.

- Scale ( $\gamma$ ) and amplitude (a) had no obvious analytical form, so they were interpolated using cubic splines.

This allowed the simulation of smooth, continuous projection curves across angles that were not originally measured, effectively increasing angular resolution without additional data collection.

![alt text](https://github.com/CamelSal/PET_Reconstruction-Enhacement/blob/master/figures/parameter_interpolation.png?raw=true)

### 4. Generating Enhanced Projections and Final Image

Using the fitted functions and interpolated parameters, new projections were synthesized for finely spaced angles. These were used to create a high-resolution simulated sinogram, which was then reconstructed using the Inverse Radon Transform to generate a final, noise-reduced image.

![alt text](https://github.com/CamelSal/PET_Reconstruction-Enhacement/blob/master/figures/enhance.png?raw=true)

## Conclusion

This project demonstrates how applying a Cauchy-based curve fit to PET projection data can significantly enhance image reconstruction. By fitting each projection with a smooth function, we were able to reduce background noise and highlight key signal features, resulting in a much cleaner and more interpretable final image.

In our case, a two-peak Cauchy distribution was used to enhance the projections. This choice was especially appropriate because the scanned object contained two distinct radioactive materials, producing two separate peaks in each projection. The Cauchy model's broad tails and peak definition made it well suited for this application.

Beyond projection enhancement, the interpolation of fit parameters across angles allowed us to simulate a continuous set of projections, improving angular resolution without collecting additional data. This process also offered valuable insight into how parameters such as peak position, amplitude, and spread behave across different angles though some of these behaviors warrant deeper investigation.

Future improvements could come from:

- Studying parameter behavior under different slit resolutions

- Exploring more complex object geometries

- Refining interpolation methods to improve simulation accuracy

A better understanding of how these parameters vary across conditions could lead to more accurate models and further enhance image reconstruction performance.




## Background
This repository was created as part of a lab investigation into image reconstruction and enhancement techniques in PET imaging.