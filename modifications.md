# Modifications

### Passing args to SketchR-2CNN

Instead of passing args via the command line, this is now done by editing the file sketchr-2cnn_options.json in the models/ directory


### Input dataset

All inputs are stored in the datasets/ directory. The input for Sketch-R2CNN needs to be named 'sketchy.pkl', and contain svg sketches in 3-point format.

The inputs for pix2pix are now simply the sketchy images. These still need to be split into three subdirectories - test, train and valid.


### Data loading

Related to the input dataset changes. The file data/aligned_dataset.py has been modified so that, instead of loading an image and dividing it into A and B categories, we just load the B image and run SketchR-2CNN to produce the sketch


### models/sketchr2pix2pix_model.py

this contains a new class ```SketchR2Pix2PixModel``` which is similar in structure to ```Pix2PixModel```, but contains an instance of ```SketchR2CNNTrain``` as an attribute. The ```SketchR2CNNTrain``` and ```BaseTrain``` classes have been modified, to add the ability to load args from json and a method to partially run SketchR2CNN and return the point feature maps as images. There is also a new method which has been added to return the RNN's trainable parameters, so they can be added to the pix2pix optimizer.

### sketch-r2pix2pix_train.py

Very similar to the pix2pix ```train.py``` script - contains the main training loop for SketchR-2pix2pix