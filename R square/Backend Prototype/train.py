import nltk
import numpy as np
import random
import json
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.stem import WordNetLemmatizer

def delete_train_file(file_name):
    file_path = "d:\\R square\\Backend Prototype\\" + file_name
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Found a file of previous training. Successfully deleted {file_name}")

delete_train_file("chatbot_model.h5")
delete_train_file("training_data.pkl")

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('d:\\R square\\Backend Prototype\\database.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Extract data from intents
words = set()
classes = set()
documents = []

for intent in intents['intents']:
    class_label = intent['tag']
    classes.add(class_label)
    
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence and lemmatize
        words.update(lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(pattern))

        # Add documents in the corpus
        documents.append((nltk.word_tokenize(pattern), class_label))

# Remove punctuation tokens
ignore_words = set(['?', '!'])
words = [word for word in words if word not in ignore_words]

# Sort classes and words
classes = sorted(classes)
words = sorted(words)

# Create training data
training = []
output_empty = [0] * len(classes)

for doc_words, doc_class in documents:
    bag = [1 if word in doc_words else 0 for word in words]

    output_row = list(output_empty)
    output_row[classes.index(doc_class)] = 1

    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Convert lists within train_x and train_y to NumPy arrays of integers
train_x = np.array([np.array(lst[0], dtype=np.int32) for lst in training])
train_y = np.array([np.array(lst[1], dtype=np.int32) for lst in training])

# Find the maximum length of lists in train_x and train_y
max_length_x = max(len(lst) for lst in train_x)
max_length_y = max(len(lst) for lst in train_y)

# Pad or truncate lists in train_x and train_y to the maximum lengths
train_x = np.array([np.pad(lst, (0, max_length_x - len(lst))) for lst in train_x])
train_y = np.array([np.pad(lst, (0, max_length_y - len(lst))) for lst in train_y])

# Build the model
model = Sequential([
    Dense(128, input_shape=(max_length_x,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(max_length_y, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=100, batch_size=8, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Save words, classes, and training data
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data.pkl', 'wb'))
