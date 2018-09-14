# LSTM+VRN Version 3 (Independent, Permuted relation map)
- The each sentence of the story are fed into the double-layer BiLSTM separately. (Consecutive sentences are NOT dependent on previous sentences).
- Processed each sentence word-by-word with an double-layer BiLSTM (with the same double-layer BiLSTM acting on each sentence independently).
- The hidden states are acquired at the end of each sentence.
- This setup invokes minimal prior knowledge, in that we delineate objects as sentences, whereas previous bAbI models processed all word tokens from all support sentences sequentially. (Santoro et al 2017)

### Relation Map
- Relation-pixel The last hidden state of each sentence is concatenated with the last hidden state of the question. 
  - *Size: (Batch, 1, Sentence Hidden Size * Question Hidden Size)*
- The relation-pixel are then concatenated with other hidden states forming 3D Tensor (Relation Map) per batch. 
  - *Size: (Batch, 4, Max Hidden Num, Sentence Hidden Size * Question Hidden Size)*
- The map will be padded to the maximum number of hidden states with permuted hidden states.

### Convolution
- Three different kernel sizes for convolving through the relation map:  
  - 4 layers of 256 filters per convolution kernels (same number of MLP, or g(x), layers that Relation Net had).
  - 1st Layer Convolution Kernel *Sizes: (2, 1), (3, 1), and (2, 2)*
    - Captures relations between two objects, three objects, and four objects.
  - 2nd-4th Layer Convolutions *Element-Wise or (1, 1)*
    - Intuition behind is to attend to different filters of each relation.
    - It also prevents from computing relations of relations.
  - 
    
### Element-Wise Convolution
- The output of the convolution will go through the element-wise convolution.
- They will be concatenated into one long vector. 
  - *Size: (Batch Size, Concatenated Length)*

### Dropout
- Dropout added after LSTM in Input and Question Module.
  - Dropout Ratio: 0.5
- Dropout added after each convolution.
  - Dropout Ratio: 0.5
  
### Linear Layer
- Three Linear Layers are used (same number of MLP, or f(x), layer that Relation Net used).
- Fully connected from the output of element-wise convolution to output classes.
  - Output classes are the number of distinct answer vocabs.