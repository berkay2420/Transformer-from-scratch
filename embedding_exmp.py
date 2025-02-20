import numpy as np

# Initialize random embeddings
embeddings = {word: np.random.rand(2) for word in ["cat", "dog", "fish", "run", "swim"] }

# Function to update embeddings
def update_embeddings(target_word, context_words, learning_rate=0.01):
  target_vector = embeddings[target_word]
  context_vectors = np.mean([embeddings[w] for w in context_words], axis=0)

  # Move target vector closer to context vectors
  embeddings[target_word] += learning_rate * (context_vectors - target_vector)

#Training Loop
for epoch in range(number_of_epochs):
  for sentence in corpus:
    for word in sentence:
      context =  get_context_words(sentence, word)
      update_embeddings(word, context)

####################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Example vocabulary and data preparation
# Assume you have a small corpus of text and a simple task like 
# predicting the next word in a sequence.

Corpus: "The cat sat on the mat. The dog sat on the rug."
Vocabulary: ["the", "cat", "sat", "on", "mat", "dog", "rug"]
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4, "dog": 5, "rug": 6}
vocab_size = len(vocab)

# Tokenize the text and create pairs of input and target words. 
# For instance, from "The cat sat on the mat", create pairs like ("The", "cat"), ("cat", "sat"), etc.
# Convert words to integer indices for processing.
input_texts = [("the", "cat"), ("cat", "sat"), ...]  # and so on
input_indices = [vocab[word] for word, _ in input_texts]
target_indices = [vocab[word] for _, word in input_texts]

# Create a Simple Neural Network
# Input Layer: The size of the input layer corresponds to the vocabulary size. 
# Use one-hot encoding for input words.
# Embedding Layer: This is a fully connected layer without activation, 
# acting as an embedding layer. The output size of this layer is the embedding size (e.g., 2 for a 2D embedding).
# Output Layer: Another fully connected layer with a softmax activation. 
# The output size is again the vocabulary size.
model = Sequential()
model.add(Dense(2, input_shape=(vocab_size,)))  # Embedding layer
model.add(Dense(vocab_size, activation='softmax'))  # Output layer

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Training
X = to_categorical(input_indices, num_classes=vocab_size)
Y = to_categorical(target_indices, num_classes=vocab_size)
model.fit(X, Y, epochs=100)

# Extract embeddings
embeddings = model.layers[0].get_weights()[0]