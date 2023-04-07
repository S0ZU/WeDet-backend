# import tensorflow as tf
# import numpy as np
# import json
# import cv2 as cv

# # Open the details.json file and load the data into a variable called "details"
# with open("details.json", "r") as file:
#     details = json.load(file)

# # Load the TensorFlow Lite model from the converted_model_5.tflite file
# interpreter = tf.lite.Interpreter(model_path="tf_lite_models/converted_model_5.tflite")

# # Allocate memory for the input and output tensors
# interpreter.allocate_tensors()

# # Get the details (shape, data type, etc.) of the input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()



# # Define a function called "predict" that takes an input image and returns the predicted class
# image = cv.imread('./f1 score test/ceylon (10).jpeg')

# def predict(input_image):
#     # Preprocess the input image by resizing it to 150x150 and normalizing the pixel values
#     to_predict_image = np.expand_dims(np.array(cv.resize(input_image, (150, 150)) / 255.0, dtype=np.float32), axis=0)
    
#     # Set the input tensor to the preprocessed image
#     interpreter.set_tensor(input_details[0]["index"], to_predict_image)
    
#     # Run the prediction by invoking the interpreter
#     interpreter.invoke()
    
#     # Get the output tensor and extract the predicted class
#     output_data = interpreter.get_tensor(output_details[0]["index"])
#     predicted_class = np.argmax(output_data[0])
    
#     # Define a dictionary that maps predicted classes to text files containing information about the plants
#     class_files = {
#         details["details"]["id"]["Ceylon spinach"]: details["details"]["information"]["Ceylon spinach"],
#         details["details"]["id"]["Goose foot"]: details["details"]["information"]["Goose foot"],
#         details["details"]["id"]["Shepherd Purse"]: details["details"]["information"]["Shepherd Purse"]
#     }
    
#     # If the predicted class is in the dictionary, print the information about the plant
#     if predicted_class in class_files:
#         if predicted_class == 0:
#             print("Not a weed")
#             # can get out put from here
#         else:
#             with open(class_files[predicted_class], "r") as f:
#                 text = f.read()
#                 return (text)

# def execute(image_str):
#     # Extract the base64 data from the string
#     image_data = base64.b64decode(image_str)

#     # Convert the image data to a NumPy array using OpenCV
#     img_array = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     data = predict(img)
#     return data
