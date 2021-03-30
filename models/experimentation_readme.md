## Options / parameters to experiment with

- Number of Sketch-R2CNN output channels (pix2pix input channels). Ideally will try values in range 1-10 channels, or at least 1 channel, 3 channel and 8 channel images.

- Sketch cropping. We can either crop the sketch svg files (which also stretches them out of proportion slightly) to fill the full dimensions of the sketch, or leave them as they have been drawn in the Sketchy dataset.

- Sketch-R2CNN inverting intensities. Compare inverting intensities vs leaving them as is - although currently Sketch-R2CNN is receiving very small gradients, which implies that it may struggle to render good sketches without the intensities inverted (if other changes are not made).

- Number of discriminator layers. The default value is 3, but increasing this to 8 appears to have helped fix the mode collapse issue where the discriminator was too weak. Will try other values, e.g. 6 layers, 10 layers.

- Number of epochs - current model looks like it may benefit from training for > 100 epochs.


## Other runs to do

- pix2pix on the sketchy **images** - as a baseline for comparison.

## Changes to make

- Introduce a modified [diversity loss](https://openreview.net/pdf?id=rJliMh09F7) to the generator, which compares the difference between an image generated from random noise, and an image generated from a sketch and aims to maximise the diversity between the two.
