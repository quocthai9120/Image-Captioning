# Image Captioning Project
Authors: Thai Hoang, Van Quach

# Abstract
In today's world, the interaction of images and text can be used to accomplish a variety of tasks. On the one hand, adding text to images can make them considerably more readable for both humans and computers. Image captioning, which is described as the task of automatically creating written descriptions for images, could help to improve this experience. Because it necessitates tackling the challenge of determining items inside an image as well as conveying the relationships between those things in natural language, it combines expertise of both computer vision and natural language processing. Image captioning is also thought to aid in the development of assistive devices that remove technological hurdles for visually impaired persons.

# Related Work
There have been several models designed to extract patterns from photos throughout history. The Convolutional Neural Network [1] is one of the major phases in extracting features from images as it allows the model to collect local information. 

A big move from the past is transfer learning., Transfer learning allows us to apply pre-trained models for our specific purpose [2]. We use the pretrained model ResNet-101 [3] in our study because it has demonstrated its capacity to perform numerous vision-related tasks.

In terms of language processing, recurrent models demonstrate the ability to cope with sequences of varying lengths. LSTM is a good option to mention since it deals with both long-term and short-term dependencies within a sequence, making it a good model for extracting information from languages for a variety of purposes, including language generation [4].

Finally, the encoder-decoder mechanism [5] enables the ability to connect distinct parts together to form a greater task. The release of the encoder-decoder techniques opens the path for a lots of tasks, including connecting "vision" and "languages" together.

# Dataset
In our project, we used the "Common Objects in Context" (COCO) 2014 dataset for images and Karpathy's split for captions.

The COCO dataset contains 164K images, which were divided into training data, validation data, and testing data as 83K/41K/41K. Each image is a three-channel RGB image of an object from one of 91 subcategories. categories. The dataset can be downloaded at: [https://cocodataset.org/#download].

As for captions, we used Karpathy's split, as it is a better format than the original COCO captions. This split was made by Karpathy and Li (2015), which divided the COCO 2014 validation data into new validation and test sets of 5000 images, as well as a "restval" set that contained the remaining approximately 30k images. Annotations are present on every split. Therefore, the .json file we got from Karpathy's split will act as "annotations". The "annotations" portion can be downloaded here: [http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip]

# Experiment
## Data Preprocessing
The data preprocessing procedure would be described as follow:
1. Get the images and captions from the source
2. Map all captions with the corresponding "image paths". A note here is that we use "image paths" instead of original images. This is because from experimenting, we notice that loading the image directly even though could boost our efficiency in term of data loading, however, it would consume a huge portion of memory, which does not worth it. To deal with the problem of image size, we decided to store the paths of images and use the path as a representation of the image. Only when we call "getitem" (which is expected to be called within a particular batch during training and evaluating), we would load the image from the provided path. A tradeoff here is that it takes us more time during training and evaluating to load the images, however, we can save a lot of memory for our other tasks (such as increasing model complexity).
3. Create a torch.Dataset object to store the images with the corresponding captions. Note here that each image has around 5 captions, so we decide to each pair of image-caption independently. That mean, when we call "getitem" to get an instance from our dataset, we would get a caption with its corresponding image.

By following the steps described above, we have our data stored in the torch.Dataset objects for training and evaluating.
## Model Architecture
In our project, we utilize the encoder-decoder mechanism to extract the features from images with encoder block, then "decode" the images features with text with decoder block. An overal view of our moodel is shown below:
![Model architecture](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/Model-architecture.png?raw=true)

### Encoder
For encoder, we use a block of pretrained ResNet-101 with its linear and pool layers removed (as we do not need the classification task) followed by a adaptive pool layer to resize our latent features to a particular size. Using this architecture, our encoder would outputs "images" of dimension (2048, 14, 14).

### Decoder
For decoder, we use "LSTM with Attention" blocks to learn the languages and learn which part of the image should the model focus for each token. Particularly, from the encoder, we receive the feature extraction of dimension (2048, 14, 14). We then transform the image to the dimension of our hidden layer and concat the transformed feature with the embedding of our "\<start\>" token. Then, for each step of decoding, we do as follow: (1) Get the embedding of the next token; (2) Compute the attention weight encoding between the image and the token to see which part of the image should the model focus on to generate the word; (3) Pass the concatenated result to our LSTM cell; (4) Predict the next token from our LSTM cell (note that each LSTM cell would receive information from the input concatenation and the previous LSTM cell).

## Training & Evaluating
    
## Text generating

# Concepts
- Encoder-Decoder architecture: An Encoder is used to encode the input into a fixed form that the machine can use and an Decoder is used to decode that input, word by word, into a sequence.
- Attention: is an interface that connects the encoder and decoder. With this technique, the model is able to selectively focus on valuable parts (i.e. pixels) of the input (i.e. image) and learn the association between features within an image.
- Beam Search: is a heuristic search algorithm. It is usually used in NLP as a final decision-making layer to choose the best output given the target variables, such as choosing the maximum probability or the next output character.
- BLEU (BiLingual Evaluation Understudy): BLEU can be seen as the most widely used evaluation indicator; however, its initial intention of the design was not for the image captioning problem, but machine translation problems. BLEU is used to analyze the correlation of n-gram between the to-be-evaluated translation statement and the reference translation statement.

# Experiments
Our final model, similarly to mentioned in (*), is a Convolutional Neural Network - Recurrent Neural Network (RNN - CNN) framework with Attention technique and Beam Search to solve the task of image captioning, which integrates both computer vision and natural language processing. Finally, we evaluated our model using BLEU scores.

## Pre-processing Data
The first step is cleaning the data, because we want to make out data be in the right format to fit in our model.

### Cleaning Data
We decided to only have a portion of the dataset due to time constraint and the lack of CPU memory. To be specific, we reduced the size of the training dataset to be 10,000 images, the validation dataset to be 1,000 images, and the testing dataset to be 2,500 images.

### Image Pre-processing
Since we are using a pretrained Encoder, here is the pretrained ImageNet ResNet-101 on PyTorch, pixel values must be in the range `[0, 1]`. The images fed to the model should be a `Float` tensor and should also be normalized by the mean and standard deviation of the ImageNet images' RGB channels.
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### Captions Pre-processing
- Format: `<start>`[`caption right here`]`<end>` because we want the decoder to know when to begin and to end the decoding process of a sentence. Since we pass the captions around as fixed size Tensors, we need to pad captions (which are naturally of varying length) to the same length with `<pad>` tokens.
- Processing format: captions fed to the model must be an `Int` tensor of dimension `N, L` where `L` is the padded length.
- Tokenization: We created a dictionary that maps all unique words to a numberical index. Thus, every word we come across will have a corresponding integer value that can be found in this dictionary.

## Model Details
### Model Baseline
We used the "CNN Encoder and Attention RNN Decoder with Beam Search" as our baseline model. The optimizer that we used was Stochastic Gradient Descent (SGD) optimizer. 

### Encoder
The Encoder is a CNN model. Its goal is to encode the input image with 3 color channels into a smaller image with "learned" channels. In our project, we used the pretrained ResNet-101 model, but we removed the last 2 layers of it (which are the linear and pool layers) since we are not doing classification.

![encoder.png](./images/encoder.png)

### Decoder
The Decoder is an RNN (LSTM) model. Its goal is to look at the encoded image and generate a caption word by word.

Normally, without Attention mechanism, we could average the encoded image across all pixels. The resulting image of this can then be fed into the Decoder as its first hidden state, with or without a linear transformation, and the caption would be generated from each predicted word. Here, a weighted average is used instead of the simple average, with the weights of the important pixels being higher. This weighted representation of the image can be concatenated with the word that preceded it at each step to produce the next one.

However, with Attention mechanism, we would like the Decoder to focus on different parts of the image at different points in the sequence. For instance, imagine that we want to generate the word `football` in `a man holds a football` then the Decoder would know that it should focus on the football itself.

![decoder_att.png](./images/decoder_att.png)

### Attention
As mentioned before, the Attention is used to compute weights. The algorithm takes into account all the sequences generated thus far, and then proceeds to describe the next part of an image.

![att.png](./images/att.png)

In this project, we used Soft Attention, where the weights of the pixels add up to 1. If there are `P` pixels in our encoded image, then at each timestep `t`.

<p align="center">
<img src="./images/weights.png">
</p>

In the context of text, it refers to the capability of the model to assign more importance to certain words within a document relative to others. For example, if we are reading a document and have to answer a question based on it, concentrating on certain tokens in the document might help you answer the question better, than to just read each token as if it were equally important.

To sum up, the entire process of this step can be described as computing the probability that a pixel is the place to look to generate the next word.

### Final Model

![model.png](./images/model.png)
The general process of each step in building the model is as follows:
- Once the Encoder finishes generating the encoded image, some types of transformation will be performed to create the initial hidden state `h` (and cell state `C`) for the RNN (LSTM) Decoder.
- At each Decoder step:
    - The encoded image along with the previous hidden state is used to generate weights for each pixel in the Attention network.
    - The previously generated word and the weighted average of the encoding are then fed to the LSTM Decoder to continue generate the next word.

### Beam Search
We do not want to decide until we've finished decoding completely since we want to choose the sequence with the highest overall score. Thus, we would like to use Beam Search to assist us in implementing this purpose.

The general process of Beam Search is as follows:
- At the first decoding step, consider the top `k` candidates.
- Generate `k` second words for each of those `k` first words.
- Choose the top `k` [first word, second word] combinations considering additive scores.
- For each of these `k` second words, choose `k` third words and choose the top `k` [first word, second word, third word] combinations.
- Repeat at each decoding step.
- After `k` sequences terminate, choose the sequence with the highest overall score.

![beam_search.png](./images/beam_search.png)
Some sequences (striked out) may fail early, as they don't make it to the top k at the next step. Once k sequences (underlined) generate the <end> token, we choose the one with the highest score.

# Training
We train our model using Google Collaborative GPU.
## Hyperparameters:
- Optimizer: Adam optimizer.
- Criterion: Cross-Entropy loss.
## Training Result:
After training over 50 epochs, our result are as follows:
### Train Loss
<p align="center">
<img src="./images/train_loss.png">
</p>

### Train Accuracy
<p align="center">
<img src="./images/train_accuracy.png">
</p>

### Val Loss
<p align="center">
<img src="./images/val_loss.png">
</p>

### Val Accuracy
<p align="center">
<img src="./images/val_accuracy.png">
</p>

# Evaluation
We used BLEU4 score to evaluate our model. After 50 epochs, our BLEU4 score ended up to be 0.1559.
<p align="center">
<img src="./images/bleu4_score.png">
</p>

# Demo
![plane_demo.png](./images/demo_plane.png)

![demo2.png](./images/demo2.png)

![demo1.png](./images/demo1.png)

![demo0.png](./images/demo0.png)

# Conclusion


# Video Description
The video description of this project can be found here.

# References
[1] Gu, J., Wang, Z., Kuen, J., Ma, L., Shahroudy, A., Shuai, B., ... & Chen, T. (2018). Recent advances in convolutional neural networks. Pattern Recognition, 77, 354-377.
[2] Zhuang, F., Qi, Z., Duan, K., Xi, D., Zhu, Y., Zhu, H., ... & He, Q. (2020). A comprehensive survey on transfer learning. Proceedings of the IEEE, 109(1), 43-76.
[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[4] Sepp Hochreiter, Jürgen Schmidhuber; Long Short-Term Memory. Neural Comput 1997; 9 (8): 1735–1780. doi: https://doi.org/10.1162/neco.1997.9.8.1735.
[5] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27.
