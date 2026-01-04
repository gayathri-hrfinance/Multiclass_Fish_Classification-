# Multiclass_Fish_Classification-
mini project with VSCode and output shown in streamlit
🐟 Multiclass Fish Image Classification
A simple deep learning project to classify fish images
📌 Project Summary
This project is about building a deep learning model that can look at a fish image and predict which species it belongs to. I trained three different models, compared their performance, and created a Streamlit app where users can upload an image and get a prediction.
Step by Step Process 
This section explains the entire project in a simple, easy to follow way.
1️⃣ Collecting the Dataset
•	I downloaded a dataset of fish images.
•	Each fish species had its own folder.
•	Example:
animal_fish/
animal_fish_bass/
fish_sea_food_shrimp/
...
2️⃣ Loading the Dataset
•	I used ImageDataGenerator from TensorFlow to load the images.
•	I split the data into:
o	Training set
o	Validation set
o	Test set
•	I applied data augmentation to improve model performance.
3️⃣ Preprocessing the Images
•	All images were resized to 224 × 224 pixels.
•	Pixel values were normalized to the range 0–1.
•	Augmentation included:
o	Rotation
o	Zoom
o	Horizontal flip
o	Width/height shift
4️⃣ Training Three Models
I trained three different models to compare their performance:
✔ A. Simple CNN (built from scratch)
•	Several convolution layers
•	MaxPooling
•	Dense layers
•	Softmax output
✔ B. VGG16 (Transfer Learning)
•	Loaded pre trained ImageNet weights
•	Removed the top layers
•	Added my own classifier
•	Fine tuned the model
✔ C. ResNet50 (Transfer Learning)
•	Same process as VGG16
•	Fine tuned on the fish dataset
5️⃣ Evaluating the Models
For each model, I checked:
•	Accuracy
•	Precision
•	Recall
•	F1 score
•	Confusion matrix
•	Training curves (loss & accuracy)
I compared all three models to find the best one.
6️⃣ Saving the Best Model
The model with the highest accuracy was saved as:
Fish_model.h5
I also saved the class label mapping:
class_indices.json
7️⃣ Building the Streamlit App
I created a simple Streamlit app that:
•	Lets the user upload an image
•	Shows the uploaded image
•	Loads the saved model
•	Predicts the fish species
•	Displays:
o	Predicted class
o	Confidence score
o	Class probabilities
8️⃣ Running the App
To run the app:
python -m streamlit run Fish_stream.py
The app opens in the browser and is ready for predictions.
9️⃣ Testing the App
•	I tested the app with images from the dataset
•	I also tested with images downloaded from the internet

