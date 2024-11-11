# Feature extraction module.

from os import listdir, curdir
from os.path import isfile, isdir, join
from random import shuffle

import cv2
import mediapipe as mp
import pickle
import sys

from validator import get_classifier

# Media pipe utilitarians
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# OpenCV text writes constants
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (255, 255, 255)
thickness = 6
lineType = 2
org = (40, 85)
X, Y, W, H = 0, 0, 125, 125

# Feature extraction limit
MAX_IMAGES_PER_CLASS = 7500

def format_vector(landmarks):
    """
    Function that flattens hand landmarks into the desired format
    :param landmarks: hand pose landmarks
    :return: features vector
    """
    vector = []
    for landmark in landmarks:
        vector += [landmark.x, landmark.y, landmark.z]
    return vector


def video_feature_extraction(classifier):
    """
    Procedure that identifies the hand pose and classifies it using the ASL alphabet
    :param classifier: trained classifier object
    """

    # Init video capture of opencv
    cap = cv2.VideoCapture(0)

    # Init hands recognizer from mediapipe
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            # Read camera image
            success, image = cap.read()

            # If read not successful ignore
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process image and extract hand position
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            # Make letter prediction using classifier
            if results.multi_hand_landmarks:
                predictions = classifier.predict([format_vector(results.multi_hand_landmarks[0].landmark)])

                # Draw black background rectangle
                cv2.rectangle(image, (X, X), (X + W, Y + H), (0, 0, 0), -1)

                cv2.putText(
                    image,
                    predictions[0],
                    org,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType
                )

            # Show image
            cv2.imshow('MediaPipe Hands', image)

            # If exit key is hit break loop
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


def save_features(features, save_path):
    """
    Function that saves an array of features and labels to a pickle file
    :param save_path: path in which the features will be saved
    :param features: array of features to save
    """
    with open(join(save_path, f'{MAX_IMAGES_PER_CLASS}_features_dump.pkl'), "wb") as writer:
        writer.write(pickle.dumps(features))


def draw_landmarks(image, landmarks, save_path, image_name):
    """
    Procedure that draws landmarks on the given image if they were found and saves it
    :param image: numpy ndarray containing the image data
    :param landmarks: recognized hand landmarks
    :param save_path: path where to save the image with the landmarks draw
    :param image_name: export image name
    """
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(save_path + image_name + ".png", image)


def extract_features(dataset_path, save_path):
    """
    Procedure that extracts the features of images located at the dataset_path and saves its features vectors to the save_path
    :param dataset_path: directory containing the dataset directories
    :param save_path: directory where to save the features vectors binary
    """
    features_vectors = {}

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
    ) as hands:
        # Retrieve all directories under the main directory path
        dirs = [d for d in listdir(dataset_path) if isdir(join(dataset_path, d))]

        # For each directory retrieve its files
        for directory in dirs:

            print("Current letter: " + directory)

            # Full directory path
            full_dir = join(dataset_path, directory)
            # Retrieve file names under the full directory path
            file_names = [f for f in listdir(full_dir) if isfile(join(full_dir, f))]
            shuffle(file_names)
            count = 0  # Count of specimens per class
            features_vectors[directory] = []  # Init directory class with empty list

            # For each image extract its features
            for file_name in file_names:
                # print("COUNT: " + str(count) + " FILENAME: " + file_name)
                # This is where the extraction is stopped when reached the maximum images per class
                if count > MAX_IMAGES_PER_CLASS:
                    break

                # Load image and flip it
                image = cv2.flip(cv2.imread(join(full_dir, file_name)), 1)
                # Use RGB as that's what mediapipe uses
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Retrieve media pipe results
                results = hands.process(image)
                # If the landmarks were found save its features
                if results.multi_hand_landmarks:
                    # Append the features vector to the class key in dictionary
                    features_vectors[directory].append(format_vector(results.multi_hand_landmarks[0].landmark))
                    count += 1  # Count new specimen

        # Save features to pickle binary file
        save_features(features_vectors, save_path)


# Example  run: python ./feature_extraction.py ./data ./features
if __name__ == "__main__":
    # This function will only be run if the current file is run directly
    if len(sys.argv) < 3:
        raise Exception("Missing machine code argument.")
    extract_features(sys.argv[1], sys.argv[2])
