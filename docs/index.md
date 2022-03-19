# Image Captioning Project
Authors: Thai Hoang, Van Quach

# Abstract
In today's world, the interaction of images and text can be used to accomplish a variety of tasks. On the one hand, adding text to images can make them considerably more readable for both humans and computers. Image captioning, which is described as the task of automatically creating written descriptions for images, could help to improve this experience. Because it necessitates tackling the challenge of determining items inside an image as well as conveying the relationships between those things in natural language, it combines expertise of both computer vision and natural language processing. Image captioning is also thought to aid in the development of assistive devices that remove technological hurdles for visually impaired persons.

# Related Work
There have been several models designed to extract patterns from photos throughout history. The Convolutional Neural Network [1] is one of the major phases in extracting features from images as it allows the model to collect local information. 

A big move from the past is transfer learning. Transfer learning allows us to apply pre-trained models for our specific purpose [2]. We use the pretrained model ResNet-101 [3] in our study because it has demonstrated its capacity to perform numerous vision-related tasks.

In terms of language processing, recurrent models demonstrate the ability to cope with sequences of varying lengths. LSTM is a good option to mention since it deals with both long-term and short-term dependencies within a sequence, making it a good model for extracting information from languages for a variety of purposes, including language generation [4].

Finally, the encoder-decoder mechanism [5] enables the ability to connect distinct parts together to form a greater task. The release of the encoder-decoder techniques opens the path for a lots of tasks, including connecting "vision" and "languages" together.

# Dataset
In our project, we used the "Common Objects in Context" (COCO) 2014 dataset for images and Karpathy's split for captions.

The COCO dataset contains 164K images, which were divided into training data, validation data, and testing data as 83K/41K/41K. Each image is a three-channel RGB image of an object from one of 91 categories. The dataset can be downloaded at: [https://cocodataset.org/#download](https://cocodataset.org/#download).

As for captions, we used Karpathy's split, as it is a better format than the original COCO captions. This split was made by Karpathy and Li (2015), which divided the COCO 2014 validation data into new validation and test sets of 5000 images, as well as a "restval" set that contained the remaining approximately 30k images. Annotations are present on every split. Therefore, the .json file we got from Karpathy's split will act as "annotations". The "annotations" portion can be downloaded here: [http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

# Experiment
## Data Preprocessing
For the image captioning task, we need to preprocess both visual and textual data. This part discusses about generating the dataset containing both images and corresponding captions.

Because of limitation of resources, we decided to use only 10000 image-caption pairs for training and 1000 image-caption pairs for validating. We unseen take 9000 image-caption pairs from the validating set that is not used for validating to perform a final test for our model.

The data preprocessing procedure would be described as follow:
1. Get the images and captions from the source
2. Map all captions with the corresponding "image paths". A note here is that we use "image paths" instead of original images. This is because from experimenting, we notice that loading the image directly even though could boost our efficiency in term of data loading, however, it would consume a huge portion of memory, which does not worth it. To deal with the problem of image size, we decided to store the paths of images and use the path as a representation of the image. Only when we call "getitem" (which is expected to be called within a particular batch during training and evaluating), we would load the image from the provided path. A tradeoff here is that it takes us more time during training and evaluating to load the images, however, we can save a lot of memory for our other tasks (such as increasing model complexity).
3. Create a torch.Dataset object to store the images with the corresponding captions. Note here that each image has around 5 captions, so we decide to each pair of image-caption independently. That mean, when we call "getitem" to get an instance from our dataset, we would get a caption with its corresponding image. We will discuss more detail of how we process captions and images data in the next subsections.

By following the steps described above, we have our data stored in the torch.Dataset objects for training, validating, and testing. Below are a few example from our dataset with images map with corresponding captions:
![Data Examples](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/example_images_with_captions.png?raw=true)

## Pre-process caption data:
- Add `<start>` and `<end>` tokens. We want to let the decoder learns where to start and end a particular sequence sequence, so we decide to add to our captions `<start>` and `<end>` tokens, making the format of each captioon becomes `<start>`[`caption`]`<end>`. Furthermore, as we pass the captions around as fixed size Tensors, we need to pad captions (which are naturally of varying length) to the same length with `<pad>` tokens.
- Processing format: captions fed to the model must be an `Int` tensor of dimension `N, L` where `L` is the padded length.
- Tokenization: We created a dictionary that maps all unique words to a numberical index. Thus, every word we come across will have a corresponding integer value that can be found in this dictionary.

## Image transformation & normalization:
To improve the model quality, it is reasonable to have the images have similar format. We decide to reach this goal by performing transformation and normalization to the images. Particularly, for each image, we resize it to a unique size of (256 x 256), then normalize it using the mean and standard deviation of ImageNet. Here, resizing the images to a unique size would make our model learns easier, and normalizing it using the mean and standard deviation of ImageNet is reasonable since the ImageNet dataset contains millions of images, making the mean and standard deviation a great value for normalizatioon. Below, we would include a table containing the mean and standard deviation of each channel:

|                    | Channel 1 | Channel 2 | Channel 3 |
|--------------------|-----------|-----------|-----------|
| mean               | 0.485     | 0.456     | 0.406     |
| standard deviation | 0.229     | 0.224     | 0.225     |

Note: Since we are using a pretrained Encoder, here is the pretrained ImageNet ResNet-101 on PyTorch, pixel values must be in the range `[0, 1]`, so the images fed to the model should be a `Float` tensor.

## Model Architecture
In our project, we utilize the encoder-decoder mechanism to extract the features from images with encoder block, then "decode" the images features with text with decoder block. An overal view of our moodel is shown below:
![Model architecture](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/Model-architecture.png?raw=true)

### Encoder
For encoder, we use a block of pretrained ResNet-101 with its linear and pool layers removed (as we do not need the classification task) followed by a adaptive pooling layer to resize our latent features to a particular size, in which we decided to make it becomes (2048, 14, 14). Using this architecture, our encoder would outputs "images" of dimension (2048, 14, 14). 

A note here is that even though we are resizing our images to a consistent image size, we still have an adaptive pooling layer here for future changes, enables the ability to modify and scale our project.

Since we used the pre-trained ResNet-101 model, we would not discuss more detail about this famous model. Instead, we do leave the reference to the original paper, which can be look at our reference [3].

### Decoder
For decoder, we use "LSTM with Attention" blocks to learn the languages and learn which part of the image should the model focus for each token. Particularly, from the encoder, we receive the feature extraction of dimension (2048, 14, 14). We then transform the image to the dimension of our hidden layer and concat the transformed feature with the embedding of our "\<start\>" token. Then, for each step of decoding, we do as follow: (1) Get the embedding of the next token; (2) Compute the attention weight encoding between the image and the token to see which part of the image should the model focus on to generate the word; (3) Pass the concatenated result to our LSTM cell; (4) Predict the next token from our LSTM cell (note that each LSTM cell would receive information from the input concatenation and the previous LSTM cell).

Here, the most interesting part of our decoder is how we use attention mechanism to connect textual and visual information together. The procedure is described as follow for each of the token in the sequence:
1. Transform the feature extraction of the image (from the encoder)
2. Get the previous LSTM hidden block or initialize the first LSTM hidden block to be the mean of the feature extraction
3. Compute the attention between the feature extraction and the LSTM hidden block mentioned above
4. Get the textual token, tokenize it and get its embedding
5. Concatenate the attention-aware feature and the textual embedding
6. Pass the concatenation visual-textual block into a LSTM cell to generate the next token and the next LSTM hidden block.
7. Keep doing so until generating the \<end\> token.

Below is the visualization of the procedure: 
![Attention Mechanism](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/attention_decoder.png?raw=true)

For a more detail summary of the decoder, we would put it below:
```python
DecoderWithAttention(
  (attention): Attention(
    (encoder_attn): Linear(in_features=2048, out_features=512, bias=True)
    (decoder_attn): Linear(in_features=512, out_features=512, bias=True)
    (full_attn): Linear(in_features=512, out_features=1, bias=True)
  )
  (embedding): Embedding(9490, 512)
  (dropout): Dropout(p=0.2, inplace=False)
  (decode_step): LSTMCell(2560, 512)
  (init_h): Linear(in_features=2048, out_features=512, bias=True)
  (init_c): Linear(in_features=2048, out_features=512, bias=True)
  (f_beta): Linear(in_features=512, out_features=2048, bias=True)
  (fc): Linear(in_features=512, out_features=9490, bias=True)
)
```

# Training, Evaluating, and Caption Generating
## Training
We divide our training process into 2 phases: (1) Training the decoder to get the sense of language and connection between images and languages; (2) Finetuning the full model end-to-end to get closer to our data attribute.

In both phases, we use the Adam optimizer and Cross-Entropy loss as optimizer and learning objective for our model. We train as follow to reach convergence:

| Batch size                                             	| 32   	|
|--------------------------------------------------------	|------	|
| Initial Decoder Learning rate (decoder training phase) 	| 1e-3 	|
| Number of epochs for training decoder                  	| 10   	|
| Initial Encoder Learning rate (finetune phase)         	| 1e-5 	|
| Initial Decoder Learning rate (finetune phase)         	| 1e-5 	|
| Number of epochs for training end-to-end               	| 5    	|
| Weight decay                                           	| 0.8  	|

Training as described above gives us the final model.

## Evaluating
Using the trained model and evaluating it on our unseen data, we received the following performance:
|                  	| Validation data 	| Test data 	|
|------------------	|-----------------	|-----------	|
| Final loss       	| 4.150010        	| 4.22226   	|
| Final accuracy   	| 64.80002        	| 64.1432   	|
| Final BLEU-score 	| 0.138669        	| 0.13430   	|

Below, we would also include our graphs generated throughout our training process, which can shows that we have been training until our model reaches convergence.

1. Train losses: ![train_losses.png](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/train_losses.png?raw=true)
2. Train accuracies: ![train_acc.png](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/train_acc.png?raw=true)
3. Validation losses: ![val_losses.png](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/val_losses.png?raw=true)
4. Validation accuracies: ![val_acc.png](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/val_acc.png?raw=true)
5. Validation BLEU-4 score: ![val_BLEU.png](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/val_bleu.png?raw=true)

## Caption generating
We decided to use Beam search to generate the captions with the highest sequence score. In our experiment, we decide to make `beam_size = 5`. We would keep generating tokens until we could generate the `<end>` token, then we choose the one with the highest score as our generated sequence.


# Demo
![plane_demo.png](https://github.com/quocthai9120/Image-Captioning/blob/main/docs/planes.png?raw=true)

# Conclusion




# Demo
Below, we are showing a demo of how to use our trained model for translation. Simply, you can just load the trained model and use that trained model for caption generating as the following video (click on the thumbnail to see the video): ![Demo video]().

In the video, we have done the following steps:

1. Load the trained model
2. Run the code of implementing generating caption method
3. Generating captions for a few inputted images

# Video
We also include a 3-minute long video where we explained our project. Readers can access our video here: ![Summarizing Video]().

# Acknowledgement
We used "PyTorch" as the framework for implmenting our models. All of the code are written manually with references to several of paper mentioned in the references. Particularly, we mainly refer to the paper [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf) to form and finetune our model architecture and refer to the following [GitHub repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) for some engineering aspect to parallelize our training procedure.

# References
[1] Gu, J., Wang, Z., Kuen, J., Ma, L., Shahroudy, A., Shuai, B., ... & Chen, T. (2018). Recent advances in convolutional neural networks. Pattern Recognition, 77, 354-377.
[2] Zhuang, F., Qi, Z., Duan, K., Xi, D., Zhu, Y., Zhu, H., ... & He, Q. (2020). A comprehensive survey on transfer learning. Proceedings of the IEEE, 109(1), 43-76.
[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[4] Sepp Hochreiter, Jürgen Schmidhuber; Long Short-Term Memory. Neural Comput 1997; 9 (8): 1735–1780. doi: https://doi.org/10.1162/neco.1997.9.8.1735.
[5] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27.
