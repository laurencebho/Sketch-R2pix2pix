# Modifications

### Passing args to SketchR-2CNN

Instead of passing args via the command line, this is now done by editing the file sketchr-2cnn_options.json in the models/ directory


### Input dataset

All inputs are stored in the datasets/ directory. The input for Sketch-R2CNN needs to be named 'sketchy.pkl', and contain svg sketches in 3-point format.

The inputs for pix2pix are now simply the sketchy images. These still need to be split into three subdirectories - test, train and valid.