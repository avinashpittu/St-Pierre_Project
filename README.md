# St-Pierre_Project
Project to create a synthetic image generator  to generate image and labeled image pairs of yeast cells under fluorescence microscopy.

---
title: "Yeast Cell Fluorescence Image Generator"
author: "Avinash Pittu"
date: "`r Sys.Date()`"
output: github_document
---

## Overview

This repository contains code to generate synthetic images of yeast cells under fluorescence microscopy. The cells are modeled as ovals with slight variations to simulate realistic shapes. The output includes both fluorescence images and corresponding labeled images.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `mahotas`
  - `scikit-image`
  - `matplotlib`

## Installation

To install the necessary Python packages, run:

```sh
pip install numpy mahotas scikit-image matplotlib
