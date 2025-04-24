# %%
import comet_ml
COMET_API_KEY = "Fho0NbqmvO0HuZr2yC3PqWhYy"

import torch
import torch.nn as nn
import torch.optim as optim

import mitdeeplearning as mdl

import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write

assert torch.cuda.is_available, "Not found GPU"
assert COMET_API_KEY != "", "Please insert your Comet API Key"

# %%
# Load song dataset
songs = mdl.lab1.load_training_data()
songs
# %%
print(songs[0])
# %%
mdl.lab1.play_song(songs[0])

# %%
# Join list of strong strings into a single string containing all songs
songs_joined = "\n\n".join(songs)
# Find all unique characters in string then sort
vocab = sorted(set(songs_joined))
print(f"There are {len(vocab)} unique characters in the dataset")


# %%
# Create 2 mappings from character to unique numerical index and vice versa
char2idx = {u : i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
print(idx2char)


# %%
# Vectorize the song strings
def vectorize_string(string) -> np.array:
    return np.array([char2idx[s] for s in string])

vectorized_songs = vectorize_string(songs_joined)


# %%
print('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
assert isinstance(vectorized_songs, np.ndarray)



# %%
# Create function to get batches
def get_batch(vectorized_songs, seq_length, batch_size) -> torch.Tensor: # Shape: (batch_size, seq_length)
    # Length of the vectorized songs based on 0-index system
    n = vectorized_songs.shape[0] - 1
    # Randomly choose the start site of the training sequences
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i : i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i + 1 : i + 1 + seq_length] for i in idx]

    x_batch = torch.tensor(np.array(input_batch), dtype = torch.long)
    y_batch = torch.tensor(np.array(output_batch), dtype = torch.long)

    return x_batch, y_batch
# Test to make sure the function works properly
test_args = (vectorized_songs, 10, 2)
x_batch, y_batch = get_batch(*test_args)
assert x_batch.shape == (2, 10), "x_batch shape is incorrect"
assert y_batch.shape == (2, 10), "y_batch shape is incorrect"
print("Batch function works correctly")


# %%
x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(x_batch[0], y_batch[0])):
    print("Step {:3d}".format(i))
    print(" input: {} ({:s})".format(input_idx, repr(idx2char[input_idx.item()])))
    print(" expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx.item()])))


# %%
# Build LSTM model
class LSTMModel(nn.Module):    
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Batch first: True to make sure the correct order of input
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    # Initiate hidden and cell states
    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))
    
    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            # x is the output of get batch function, x.size(0) is the batch size
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)

        out = self.fc(out)
        return out if not return_state else (out, state)

# %%
# Test with simple models
vocab_size = len(vocab)
embedding_dim = 256
hidden_size = 1024
batch_size = 8

device = torch.device("cuda")

model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)
print(model)

# %%
# Test with some sample data
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
x = x.to(device)
y = y.to(device)

pred = model(x)
print(f"Input shape: {x.shape} # (batch_size, sequence_length)")
print(f"Prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")

# %%
cross_entropy = nn.CrossEntropyLoss()
def compute_loss(labels, logits):
    # Flatten labels and logits tensors
    batched_labels = labels.view(-1)
    batched_logits = logits.view(-1, logits.size(-1))
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

# %% 
# Test with output of untrained-model
example_batch_loss = compute_loss(y, pred)

print(f"scalar_loss: {example_batch_loss.mean().item()}")

# %%
# Declare the parameters of the model in the dictionary form
params = dict(
    num_training_iterations = 3000,
    batch_size = 8,
    seq_length = 100,
    learning_rate = 5e-3,
    embedding_dim = 256,
    hidden_size = 1024
)
# Specify and create the directory to save the model-related materials
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

# %%
# Create Comet experiment to track our training run
def create_experiment():
    # end any prior experiments
    if 'experiment' in locals():
        experiment.end()
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name = "6S191_Lab1_Part2"
    )

    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()

    return experiment

# %%
model = LSTMModel(vocab_size, 
                  params['embedding_dim'],
                  params["hidden_size"])
model.to(device)
optimizer = optim.Adam(model.parameters(), 
                       lr=params['learning_rate'])

def train_step(x, y):
    model.train()
    optimizer.zero_grad()
    y_hat = model(x)
    loss = compute_loss(y, y_hat)
    loss.backward()
    optimizer.step()
    return loss

# %%
history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear()
for iter in tqdm(range(params['num_training_iterations'])):
    x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params['batch_size'])
    x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

    loss = train_step(x_batch, y_batch)

    experiment.log_metric("loss", loss.item(), step=iter)
    # Update progress plot and visualize it
    history.append(loss.item())
    plotter.plot(history)

    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)

torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()




# %%
def generate_text(model:LSTMModel, start_string, generation_length = 1000):
    input_idx = [char2idx[s] for s in start_string]
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)
    state = model.init_hidden(input_idx.size(0), device)

    text_generated = []
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state = True)
        predictions = predictions.squeeze(0)

        input_idx = torch.multinomial(torch.softmax(predictions, dim = -1), num_samples=1)
        text_generated.append(idx2char[input_idx].item())

    return (start_string + ''.join(text_generated))



# %%
generated_text = generate_text(model, 'X', 10000)

# %%
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
# %%
for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)

    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)

        numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
        wav_file_path = f"output_{i}.wav"
        write(wav_file_path, 88200, numeric_data)
        experiment.log_asset(wav_file_path)

# %%
experiment.end()


