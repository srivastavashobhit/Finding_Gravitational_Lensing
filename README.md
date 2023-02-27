# Finding Gravitational Lens

## What are Gravitational Lenses?

Gravitational Lenses are an observational phenomenon caused by the alignment of two galaxies separated by cosmological distances. The light from the aligned background galaxy bends because of gravitational forces from the foreground galaxy. This causes the light from the background galaxy to appear to us on earth like a halo or Einstein Ring as shown in the diagram and example to the right.


<p align="center">
  <img src="https://aapt.scitation.org/action/showOpenGraphArticleImage?doi=10.1119/1.5135783&id=images/medium/1.5135783.figures.online.f1.jpg" style="width:500px;"/>
</p>
<p align="center">
Fig 1. Depiction of Gravitional Lensing phenomenon
</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/11/A_Horseshoe_Einstein_Ring_from_Hubble.JPG" style="width:500px;" />
</p>
<p align="center">
Fig 2. LRG 3-757 Gravitation Lens captured by Hubble Space Telescope's Wide Field Camera 3
</p>


## Project Goal

To precisely segment existing Gravitational Lenses in Deep Space to ease and foster scientific research works.

## Results

<p align="center">
  <img src="Support_Files/Lens.png" style="width:500px;" />
</p>
<p align="center">
Fig 3. Segmentation Results on the Lens component
</p>

<p align="center">
  <img src="Support_Files/Arcs.png" style="width:500px;" />
</p>
<p align="center">
Fig 4. Segmentation Results on the Arc component
</p>

## Model Architecture

We used Nested U2-Net with Intermediate Supervision from https://github.com/xuebinqin/DIS

<p align="center">
  <img src="Support_Files/is-net.png" style="width:700px;" />
</p>



