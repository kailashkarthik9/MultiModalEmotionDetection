import matplotlib.pyplot as plt
import torch
from transformers import BertModel, BertTokenizer, BertConfig

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"

model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')

config = BertConfig()
config.output_hidden_states = True

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights, config=config)

text = "Here is the sentence I want embeddings for to give to Sarthak."
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indices.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Print out the tokens.
print(tokenized_text)

encoded_tokens = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
print(tokenizer.encode(text, add_special_tokens=True))

# Mark each token as belonging to sentence "1".
segments_ids = [0] * len(tokenized_text)

print(segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    # last_encoded_layer, _ = model(tokens_tensor, token_type_ids=segments_tensors)
    # last_encoded_layer, _ = model(encoded_tokens)
    outputs = model(encoded_tokens)
    last_hidden_state, pooler_output, hidden_states = outputs
    hidden_states = list(hidden_states)[1:]

    print("Number of batches:", len(last_hidden_state))
    batch_i = 0

    print("Number of tokens:", len(last_hidden_state[batch_i]))
    token_i = 0

    print("Number of hidden units:", len(last_hidden_state[batch_i][token_i]))

    # For the 5th token in our sentence, select its feature values from layer 5.
    token_i = 5
    layer_i = 5
    vec = hidden_states[layer_i][batch_i][token_i]

    # Plot the values as a histogram to show their distribution.
    plt.figure(figsize=(10, 10))
    plt.hist(vec, bins=200)
    plt.show()

    # # # Re-ordering the hidden states to be token-major

    # `hidden_states` is a Python list.
    print('Type of encoded_layers: ', type(hidden_states))
    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', hidden_states[0].size())
    token_embeddings = torch.stack(hidden_states, dim=0)
    print(token_embeddings.size())
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    print(token_embeddings.size())
    token_embeddings_re_ordered = token_embeddings.permute(1, 0, 2)
    print(token_embeddings_re_ordered.size())

    # # # Word Vectors - summing the last four layers

    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []
    # `token_embeddings_re_ordered` is a [22 x 12 x 768] tensor.
    # For each token in the sentence...
    for token in token_embeddings_re_ordered:
        # `token` is a [12 x 768] tensor
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    # # # Sentence Vector

    # `hidden_states` has shape [12 x 1 x 22 x 768]
    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[11][0]
    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    print("Our final sentence embedding vector of shape:", sentence_embedding.size())
    print(sentence_embedding.tolist())
