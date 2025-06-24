# PET-Reconstruction-Ennhacement

This repository contains a project focused on enhancing **Positron Emission Tomography (PET)** image reconstruction using Python-based techniques, including curve fitting and image interpolation. The goal is to reduce background noise and improve image resolution in reconstructed PET scans.

## Introduction

Positron Emission Tomography (PET) is a nuclear medicine imaging technique used to map metabolic activity in the body. It works by administering biologically active molecules labeled with positron-emitting radionuclides. When these positrons encounter electrons, they annihilate and produce two gamma photons traveling in opposite directions.

By placing detectors opposite each other, the resulting gamma rays can be detected. Measuring the difference in arrival times between many such pairs across multiple angles allows for the creation of radiation flux projections. These are combined using an **Inverse Radon Transformation** to reconstruct an image of the tracer distribution within the body.

This project builds on an understanding of the PET reconstruction process to enhance image quality and reduce noise. After evaluating the limitations of the imaging equipment, we apply a two-peak Cauchy distribution to fit each projection curve:

$$
f(x) = \frac{a}{\pi s(1+(\frac{x - x_{01}}{s})^2)} + \frac{a}{\pi s(1+(\frac{x - x_{02}}{s})^2)} + y_{\text{off}}
$$

We then analyze how these fitted parameters change with angle, enabling interpolation across angles. This allows us to simulate projections at finer angular resolutions and reconstruct a smoother, denoised image using the Inverse Radon Transform.
