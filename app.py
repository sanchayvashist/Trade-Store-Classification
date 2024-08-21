import json
import time
import pytesseract
import os
import itertools

import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

# Get response from gemini 
def get_gemini_response(model, img):
    message = []

    with open("prompt.txt", "r") as file:
        prompt = file.read()

    message.append(prompt)
    message.append(img)
    start_time = time.time()
    response = model.generate_content(message)
    end_time = time.time()
    return response.text, end_time - start_time

# Funciton to Remove Rotation from image
def straighen_image(image):
    text = pytesseract.image_to_osd(image)

    rotation_angle = 0
    for line in text.split("\n"):
        if line.startswith("Rotate:"):
            rotation_angle = int(line.split(":")[1].strip())

    return image.rotate(rotation_angle)

def print_metrics(y_true, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate recall
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# make confusion matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 

    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

def get_predictions(base_dir, result_df, model):
    y_true = []
    y_pred = []

    for root, dir, files in os.walk(base_dir):
        for file in tqdm(files):
            y_true.append(os.path.basename(root))
            filepath = os.path.join(root, file)
            img = Image.open(filepath)
            response, time_taken = get_gemini_response(model, img)
            response = response.strip("```json").strip("```").strip()
            dic = json.loads(response)
            dic["path"] = filepath
            dic["original category"] = os.path.basename(root)
            y_pred.append(dic["Category"])
            result_df.loc[result_df.shape[0]] = dic
            time.sleep(30)
        
    return y_true, y_pred

if __name__ == "__main__":
    load_dotenv()

    key = os.environ["API_KEY"]
    
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')        

    result_df = pd.DataFrame(columns=['path', "original category", "Valid Shop", "Category", "Movability", "Sign"])
    y_true, y_pred = get_predictions(base_dir="downloaded_images", result_df= result_df, model = model)
    print_metrics(y_true, y_pred)
    make_confusion_matrix(y_true, y_pred, classes= ["General Trade", "Not General Trade"],savefig=True)
    result_df.to_csv("result.csv", index= False)