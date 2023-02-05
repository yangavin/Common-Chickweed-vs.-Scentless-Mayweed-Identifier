# Common Chickweed vs. Scentless Mayweed Identifier

## Folders
* train - Contains all training data (80% of data)
* test - Contains all test data (20% of data)

## Files
* main.py - Uses various AI and image processing libraries to allow AI to learn and develop a model for classification
* model_saved.h5 - Model that's saved after learning from the dataset
* model_test.py - Uses the model saved to classify completely new images

## To Create a New Model
Simply run main.py. You can also adjust amount of epochs or other parameters

## To Classify Entirely New Images
Download and insert an image to the root folder, then change the directory for the image variable to the file name of the image