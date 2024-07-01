

import streamlit as st
import tensorflow as tf
import numpy as np

#NEED THIS TO RUN DEEP LEARNING
tf.config.set_visible_devices([], 'GPU')



test_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/HP/Documents/PYTHONCODES/DATASETS/FRUITS&VEGGIE(NOOVERSAMPLING)/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Test set for veggies classes
test_set_veggie = tf.keras.utils.image_dataset_from_directory(
    '/Users/HP/Documents/PYTHONCODES/DATASETS/FRUITS&VEGGIE(NOOVERSAMPLING)/test/Vegetables',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Test set for fruits classes
test_set_fruit = tf.keras.utils.image_dataset_from_directory(
    '/Users/HP/Documents/PYTHONCODES/DATASETS/FRUITS&VEGGIE(NOOVERSAMPLING)/test/Fruits',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

def model_prediction_fruit(prediction_image):
    cnn_fruit = tf.keras.models.load_model('CNN_Fruits.h5')
    cnn_fruit_gray = tf.keras.models.load_model('CNN_Fruit_gray.h5')

    #Preprocessing
    image = tf.keras.preprocessing.image.load_img(prediction_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    #Fruit Prediction   
    #RGB color
    predictions_rgb = cnn_fruit.predict(input_arr)

    # Convert the image to grayscale
    gray_image = tf.image.rgb_to_grayscale(input_arr[0]).numpy()

    # Get predictions from Grayscale model
    predictions_gray = cnn_fruit_gray.predict(np.array([gray_image]))

    # Combine predictions (example: simple average)
    fruit_combined_predictions = (predictions_rgb + predictions_gray) / 2.0

    return np.argmax(fruit_combined_predictions)

def model_prediction_vegetables(prediction_image):
    cnn_veggie = tf.keras.models.load_model('CNN_Veggie.h5')

    #Preprocessing
    image = tf.keras.preprocessing.image.load_img(prediction_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    predictions_veggie = cnn_veggie.predict(input_arr)

    return np.argmax(predictions_veggie)

def model_prediction_fvsv(prediction_image):
    #Models
    cnn1 = tf.keras.models.load_model('FruitVSVeggie.h5')

    #Preprocessing
    image = tf.keras.preprocessing.image.load_img(prediction_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    #Predictions
    predictions1 = cnn1.predict(input_arr)

    return np.argmax(predictions1)


st.header("Fruits & Vegetables Prediction")
prediction_image = st.file_uploader("Choose an Image")
if(st.button("Show Image")):
    st.image(prediction_image, width=4,use_column_width=True)
if(st.button("Predict")):
    st.write("Our Prediction")

    result_index = model_prediction_fvsv(prediction_image)
    result_index_veggie = model_prediction_vegetables(prediction_image)
    result_index_fruit = model_prediction_fruit(prediction_image)

    if test_set.class_names[result_index] == "Fruits":
        result_str = "It's a {} and it is {}".format(
            test_set.class_names[result_index],
            test_set_fruit.class_names[result_index_fruit]
    )
    elif test_set.class_names[result_index] == "Vegetables":
        result_str = "It's a {} and it is {}".format(
            test_set.class_names[result_index],
            test_set_veggie.class_names[result_index_veggie]
    )
    else:
        result_str = "Unrecognized category"

    st.success(result_str)