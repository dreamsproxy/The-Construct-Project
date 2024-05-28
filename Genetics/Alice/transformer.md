# Transformer explanation:
## Input Representation:
    The input sequence is represented as a set of vectors, often called embeddings. Each vector corresponds to a token in the sequence (e.g., a word or a subword).

## Queries, Keys, and Values:
    For each token in the input sequence, the self-attention mechanism creates three vectors: the query vector, the key vector, and the value vector.
    
    These vectors are derived from the input embeddings through learned linear transformations.

## Scoring:
    The query vectors are compared to the key vectors to produce a score for each pair of tokens. This score reflects how much attention one token should pay to another.
    The score is calculated using a dot product between the query and key vectors A higher dot product indicates a stronger connection.

## Attention Weights:
    The scores are scaled using a scaling factor (often the square root of the dimension of the key vectors) and then passed through a softmax function to obtain attention weights.
    
    Softmax ensures that the weights sum up to 1.

## Weighted Sum:
    The attention weights determine how much each token should "pay attention" to the others. These weights are used to compute a weighted sum of the value vectors.
    
    This weighted sum becomes the output for the token under consideration. It encapsulates information from other tokens in the sequence.

## Multi-Head Attention:
    Transformers typically employ multiple sets of queries, keys, and values, often referred to as "attention heads."
    
    This allows the model to capture different aspects and relationships within the data.

## Output:
    The outputs of the multiple attention heads are concatenated and linearly transformed to produce the final output for the token.
    
    This output is then used as input for subsequent layers in the model.