# Development and Deployment of Style Transfer Mini-Program
**This Project is Finished in Fall 2020**

**Team Members: Zhao Chengliang, Sun Yize, Wu Jincheng, Xu Hanjia, Zhan Pei**

## 1. Style Transfer Principles

**style transfer** is an image generation algorithm based on convolutional neural networks. We select a style image and a content image, and the algorithm automatically generates a new image. The specific effect is shown in the figure below;

 we select a photo of the gate of Shandong University as the content image, and Van Gogh's famous "Starry Night" as the style image. The trained algorithm automatically generates a picture of the university gate with the style of "Starry Night". Neural style transfer technology perfectly applies deep learning technology to the art field, combining the beauty of technology and art.

<img src="https://s1.ax1x.com/2020/10/05/0tMhp8.png" alt="0tMhp8.png" border="0" width=80%/>

### 1.1 Loss Function Definition

To build a neural style transfer system, we define a cost function for the generated image. By minimizing this cost function, we generate the final image. In Gatys' paper, the overall loss function is defined as follows:

$$ J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)$$

Where $J_{content}$ measures the content difference between the generated image and the content image; $J_{style}$ represents the style and texture difference between the generated image and the style image. After defining the loss function, we can continuously minimize this loss function using gradient descent.

During this process, we are actually updating each pixel in the generated image.

- **Content Loss**

In convolutional neural networks, each layer uses the output of the previous layer to further extract more complex features until they are complex enough to be used for object recognition. So each layer can be viewed as a collection of local feature extractors. We want to use these pre-trained neural networks to measure the similarity between the generated image and the content image. Generally, lower layers describe specific visual features of the image (such as texture, color, etc.), while higher layer features are more abstract descriptions of the image content. So we compare the similarity of high-layer features of two images. Specifically, we extract the results computed by a specific layer of the network for the image, which are the activation factors of this specific layer. The content loss is defined as:

$$J_{content}=\frac{1}{2}\left \| a^{[l](C)} - a^{[l](G)}\right \|$$

Where $ a^{[l](C)}$ and $ a^{[l](G)}$ represent the activation factors (in matrix form in CNN) of the content image and the generated image at the l-th layer of the neural network. Here, VGG16 or VGG19 is generally selected as the pre-trained network for feature extraction.

- **Style Loss** **Style** is an abstract concept in art, but certain shapes in paintings often have specific colors. Taking "Starry Night" as an example, circular shapes are always blue or yellow. So if an image has many blue or yellow concentric circles, we can consider it to have some features of "Starry Night". The convolutional layers in convolutional neural networks are feature extractors, and different channel convolution kernels may extract different image features, such as a specific color or shape. If we calculate the correlation between these features, if the correlation is large, it indicates that this image has more of this style. Therefore, we use the pre-trained network VGG16 to calculate the style loss. **Gram Matrix**: $$G_{k,k{}'}^{[l]}=\sum_{i=1}^{n_{H}^{[l]}}\sum_{i=1}^{n_{W}^{[l]}}a_{i,j,k}^{[l]}a_{i,j,k{}'}^{[l]}$$

Where $a_{i,j,n}^{[l]}$ is the activation value at the i-th row, j-th column of the n-th channel after the image is computed through the l-th layer of the network. The Gram matrix $G^{[l]}$ is a $n_{H}^{[l]} \times n_{W}^{[l]}$ matrix. The main function of the Gram matrix is to measure the correlation between different channels at a specific layer of the neural network. Its essence is the matrix obtained by transposing and multiplying the activation factor matrix (feature map) generated after computing each layer of the convolution kernel.

After defining the Gram matrix, we can use the Gram matrix of specific layers in the network to measure the generated image and the style image. The style loss is defined as follows:

$$J_{style}^{[l]}=\frac{1}{(2n_{H}^{[l]}n_{W}^{[l]}n_{C}^{[l]})}\sum_{k}^{}\sum_{k{}'}^{}(G_{k,k{}'}^{[l](G)}-G_{k,k{}'}^{[l](S)})^2$$

### 1.2 Original Style Transfer[1](#refer-anchor-1)

<img src="https://s1.ax1x.com/2020/10/06/0Njtit.png" alt="0Njtit.png" border="0"  width=70%/> In Gatys' paper, he used the VGG19 network as the content and style extraction network. The style loss is calculated using the parameters of Conv1_2, Conv2_2, Conv3_3, and Conv4_4 convolutional layers. The content loss is calculated based on the parameters of the Conv3_3 layer.

The original style transfer process is shown in the figure below. After selecting the style image and content image, we first randomly generate a noise image. Then for each update, we first calculate the loss function of the generated image, then use gradient descent to minimize the loss function, and update each pixel in the generated function. <img src="https://s1.ax1x.com/2020/10/05/0tw7FO.png" alt="0tw7FO.png" border="0" width=70%/>

Thus, generating each image is a "training" process. In actual testing, this process is often time-consuming. Depending on the image size and computer performance, this process can take several hours.

### 1.3 Fast Style Transfer[2](#refer-anchor-2)

Traditional style transfer algorithms often require a long time to generate an image. This is mainly because this image generation algorithm is essentially a "training" process, which requires a lot of computation and memory. So a natural idea emerged: what if we don't treat the image generation as a "training" process, but as a computational process. In Johnson's paper, he first proposed the concept of fast style transfer. It mainly generates images through a convolutional neural network and uses VGG16 to calculate style loss and content loss. Then it updates the parameters of the image generation network through gradient descent. The specific process is shown in the figure below:

<img src="https://s1.ax1x.com/2020/10/06/0U2oDO.png" alt="0U2oDO.png" border="0" width=70%/>

The above is the image from the paper. The entire training and generation process includes two networks, the **Image Generation Network** and the **Loss Network**.

<img src="https://s1.ax1x.com/2020/10/06/0UR5Js.png" alt="0UR5Js.png" border="0"  width=70%/>

- **Image Generation Network** The image transformation network is also a residual network. It consists of 3 convolutional layers, 5 residual blocks, and 3 convolutional layers. No pooling operations are used for sampling here. In the initial convolutional layers (second and third layers), downsampling is performed, and in the last 3 convolutional layers, upsampling is performed to restore the image to a size of 256*256, while normalizing the values to (1, 255) to represent RGB colors, which will then be input to the VGG16 network for calculation.

<img src="https://s1.ax1x.com/2020/10/07/0au9II.png" alt="0au9II.png" border="0" width=70% />

- **Loss Network**

The main function of the loss network is to calculate style loss and content loss. In each training session, we use a photo from the training set as the content image. First, we generate a new image through the image generation network, then pass the **content image** and **generated image** through VGG16 calculation, and extract their activation values in the Conv2-2 layer to calculate the content loss. Similarly, we use the activation values of the **style image** and **generated image** in Conv1-2, Conv2-2, Conv3-3, and Conv4-4 to calculate the style loss. <img src="https://s1.ax1x.com/2020/10/07/0aVT9U.png" alt="0aVT9U.png" border="0" width=70% />

- **Training Process**

During the training process, we keep one style image fixed and select images from the training set as content images. The training set uses the public MSCOCO2014 dataset, which contains about 580,000 images. This allows the algorithm to fully learn the style from the style image.

### 1.4 Comparison

**The definition of the loss function is the same for both original style transfer and fast style transfer.**

| **Algorithm**               | Network Training Required                       | Pre-trained Network | Image Generation Method                                      | **Generation Time** |      |
| --------------------------- | ----------------------------------------------- | ------------------- | ------------------------------------------------------------ | ------------------- | ---- |
| **Original Style Transfer** | No                                              | VGG19               | Minimizes the generation function through gradient descent.<br>Directly generates images | Several hours       |      |
| **Fast Style Transfer**     | Yes, needs to train an image generation network | VGG16               | Inputs the content image into the network,<br>generates a new image after calculation | Several seconds     |      |

**Because the style image is fixed during the training of fast style transfer, this means that each model corresponds to only one style. So to generate images in multiple styles, we need to train multiple models.**

- **Download and Read VGG16 Pre-trained Network** Address: 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

## 2. Fast Style Transfer Implementation

**Using Keras Deep Learning Framework for Construction and Training**

### 2.1 Generation Network and Loss Network

#### 2.1.1 Image Generation Network

The specific structure of the image generation network is given by Professor Li Fei-Fei in her paper[3](#refer-anchor-3) <img src="https://s1.ax1x.com/2020/10/07/0dkxiT.png" alt="0dkxiT.png" border="0" width=70%/>

The specific structure of the residual block is as follows:

<img src="https://s1.ax1x.com/2020/10/07/0dEZBn.png" alt="0dEZBn.png" border="0" width=70%/> The left side is the residual block used in this network, and the right side is the normal one.

- **Keras Custom Layers** In the fast style transfer task, to facilitate programming, we need to add some custom layers to the network (such as normalization processing). The code is defined in the form of classes, and needs to be strictly implemented according to the Keras manual.

python

```python
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Deconvolution2D,  Conv2D,UpSampling2D,Cropping2D,Conv2DTranspose
from keras.layers import add
from keras.engine.input_layer import Input
```

python

```python
class Input_Norm(Layer):# Normalize to make image matrix values between (0,1)
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self,input_shape):
        return input_shape

    def call(self, x, mask=None):
        return x/255.
class Denorm(Layer):# Inverse normalization to transform results distributed between (-1,1) to (0,255)

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass
    def compute_output_shape(self,input_shape):
        return input_shape
    def call(self, x, mask=None):
        return (x + 1) * 127.5

def res_block(x):
    y = x
    x = Conv2D(128,kernel_size = (3,3),strides = (1,1),padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128,kernel_size = (3,3),strides = (1,1),padding = 'same')(x)
    x = BatchNormalization()(x)
    res = add([x, y])
    return res
def img_transform_net():
    input = Input(shape=(256,256,3))
    input = Input_Norm()(input)
    #First convolutional layer
    x =Conv2D(32, kernel_size = (9,9), strides = (1,1), padding = 'same')(input)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #Second convolutional layer
    x =Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #Third convolutional layer
    x =Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #Residual network
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    # Upsampling
    x =Conv2DTranspose(64, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #Upsampling
    x =Conv2DTranspose(32, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #Upsampling
    x =Conv2DTranspose(3, kernel_size = (9,9), strides = (1,1), padding = 'same')(x)
    x =BatchNormalization()(x)
    output = layers.Activation('tanh')(x)  
    # Inverse normalization
    output= Denormalize()(output)
    #Define model
    model = Model(inputs = input,outputs = output)  
    return model
```

#### 2.1.2 Loss Network

- **Gram Matrix**

```python
def gram_matrix(x):#Reference Keras example
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
```

- **Style Loss and Content Loss**

In this process, we need to customize the loss function. The custom loss layer is added to the model as a network layer, and the output of this loss is used as the objective function for network optimization.

```python
from keras import backend as K
from keras.regularizers import Regularizer
from keras.objectives import mean_squared_error

class Style_loss(Regularizer):# Calculate the style loss of a certain layer

    def __init__(self, style_feature_target, weight=1.0):
        self.style_feature_target = style_feature_target
        self.weight = weight
        self.uses_learning_phase = False
        super(StyleReconstructionRegularizer, self).__init__()

        self.style_gram = gram_matrix(style_feature_target)# Define according to the manual

    def __call__(self, x):
        output = x.output[0]
        style_loss = self.weight *  K.sum(K.mean(K.square((self.style_gram-gram_matrix(output) )))) 

        return style_loss
class Content_loss(Regularizer):#Calculate the content loss of a certain row
    def __init__(self, weight=1.0):
        self.weight = weight
        self.uses_learning_phase = False
        super(FeatureReconstructionRegularizer, self).__init__()

    def __call__(self, x):
        generated = x.output[0] 
        content = x.output[1] 

        loss = self.weight *  K.sum(K.mean(K.square(content-generated)))
        return loss
```

```python
from PIL import Image
import numpy as np
def Image_PreProcessing(path,img_width, img_height):
# Storage path of image to be processed	
    im = Image.open(path)
    
    imBackground = im.resize((img_width, img_height))
   
    re_img = np.asarray(imBackground)
    return re_img
```

```python
def compute_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict,weight):
    style_img = Image_PreProcessing(style_image_path, 256, 256)
    

    style_loss_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']# Calculate from VGG16

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-19].input], style_layer_outputs)

    style_features = vgg_style_func([style_img])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]

        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=weight)(layer)

        layer.add_loss(style_loss)

def compute_content_loss(vgg_layers,vgg_output_dict,weight):

    content_layer = 'block4_conv2'
    content_layer_output = vgg_output_dict[content_layer]

    layer = vgg_layers[content_layer]
    content_regularizer = FeatureReconstructionRegularizer(weight)(layer)
    layer.add_loss(content_regularizer)
```

python

```python
from keras.engine.topology import Layer

class VGG_Norm(Layer):# Normalize before the image enters the loss network
    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)
    def build(self, input_shape):# Required by the manual, no actual meaning
        pass
    def call(self, x):
        x = x[:, :, :, ::-1]       
        x = x-120
        return x
    
def dummy_loss(y_true, y_pred ):# Comprehensive loss function
    return K.variable(0.0)
```

python

```python
from keras.applications.vgg16 import VGG16
from keras.layers.merge import concatenate
transfer_model = img_transform_net()

tensor = concatenate([transformer.output,transformer.input],axis = 0)
tensor = VGGNormalize(name="vgg_normalize")(tensor)

vgg = VGG16(include_top = False,input_tensor = tensor2,weights = None)
vgg.load_weights("/Users/zcl271828/Downloads/fst-2/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",by_name = True)

vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-18:]])
vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-18:]])
```

- Combine the generation network and the loss network to prepare for final training

python

```python
add_style_loss(vgg,'images/style/starry_night.jpg' , vgg_layers, vgg_output_dict,3)   
add_content_loss(vgg_layers,vgg_output_dict,2)
for layer in vgg.layers[-19:]:
    layer.trainable = False#Fix loss network parameters
```

### 2.2 Training Dataset

#### 2.2.1 Download and Read Dataset

The COCO dataset is a large, rich object detection, segmentation, and captioning dataset. This dataset aims at scene understanding, mainly taken from complex everyday scenes. Download address: http://images.cocodataset.org/zips/train2014.zip

Below is an image from the training set:

```python
#
from matplotlib.pyplot import imshow
imshow(Image_PreProcessing("/Users/zcl271828/train2014/COCO_train2014_000000291797.jpg",256,256))
```

<img src="https://s1.ax1x.com/2020/10/07/0dcsQP.png" alt="0dcsQP.png" border="0" />

### 2.3 Training Process

#### 2.3.1 Determine Model Parameters

Due to limited computational resources, we are unable to conduct large-scale experiments to find the optimal hyperparameters. Therefore, we determined the optimal hyperparameters by reviewing literature:

- **batch size=1**
- **learning rate=1e-3**
- **training iterations=83000**

#### 2.3.2 Conduct Training

python

```python
from keras.preprocessing.image import ImageDataGenerator

optimizer = keras.optimizers.Adam()#Using Adam optimizer for training with default hyperparameters to effectively avoid overfitting
vgg1.compile(optimizer,loss=dummy_loss)
gen = ImageDataGenerator()
train_data = gen.flow_from_directory('/fst-2/image/training_data',batch_size = 1,target_size = (256,256),class_mode = 'input')
# Keras API for image training to automatically generate training data
history= vgg.fit_generator(train_data,steps_per_epoch=83000,epochs=1)
vgg.save_weights('starry_night_weights.h5')
vgg.save('starry_night.h5')# Save model
```

#### 2.3.3 Training Process

Due to limited personal computer computing resources, we found a GPU server for deep learning online. So we rented a deep learning server for 3 yuan/hour. The specific configuration is GTX1080 TI. Each model training took about 5-6 hours, and a total of 7 models were trained. Below is a screenshot of the console: <img src="https://s1.ax1x.com/2020/10/07/0dBGMF.png" alt="0dBGMF.png" border="0"  width=60% />

After training was completed, we downloaded the model files to our local machine, and then uploaded them to our own server. This prepared for the server to read the model and generate images in the next step.

# 3. Cloud Server API Setup

## 3. Cloud Server API Setup

In the process of deploying the model, considering that the trained model is large and requires computational devices with higher computing power, we ultimately chose to set up the model on a cloud server rather than deploying it directly on the WeChat mini-program. This ensures that the model can calculate in a shorter time and has enough memory space for model operation.

### 3.1 Server Type Comparison

During the neural style transfer model computation process, there are high demands on device memory and CPU. Using TensorFlow will occupy a large amount of memory space, which requires us to fully consider the model's requirements when purchasing a server.

Currently, based on cost and actual needs, we can choose between **general-purpose servers** and **burstable servers**.

General-purpose cloud servers (ECS) are suitable for high network packet sending and receiving scenarios, such as video bullet screens, telecommunications business forwarding, enterprise applications, and other application scenarios.

Burstable servers are suitable for web application servers, development testing and pressure testing applications, and are not suitable for scenarios that require performance above the "baseline" for long periods or enterprise computing performance requirements.

| Server                 | Host Configuration | CPU Performance | Price          |
| ---------------------- | ------------------ | --------------- | -------------- |
| General-purpose Server | 2 cores 4G         | 100% usage      | 248 yuan/month |
| Burstable Server       | 2 cores 4G         | 15% usage       | 54 yuan/month  |

Considering that when using the server, we will only occupy it when processing images, and the number of images processed is limited, with the server being idle at other times. Therefore, we finally decided to use a burstable server as our cloud server for style transfer.

### 3.2 Setting up the Flask Framework

Since all the project code is implemented in Python, we chose Python's Flask framework when building the web framework.

Flask first receives the image sent by the WeChat mini-program and stores it on the cloud server; then it calls the style transfer model to calculate the image and stores the calculated image; finally, Flask sends the generated image back to the WeChat mini-program in the form of a return value.

**The following code runs on our own server, interface: https://experimentforzcl.cn:8080**

#### 3.2.1 Read the Model and Generate Images

python

```python
from imageio import imwrite,imread
import numpy as np
from PIL import Image
import nets
import os
import string
import random
import json
import requests
from flask import Flask, request, redirect, url_for, render_template,send_file
import base64
# Crop image
def crop_image(img):
    aspect_ratio = img.shape[1]/img.shape[0]
    if aspect_ratio >1:
        w = img.shape[0]
        h = int(w // aspect_ratio)
        img =  K.eval(tf.image.crop_to_bounding_box(img, (w-h)//2,0,h,w))
    else:
        h = img.shape[1]
        w = int(h // aspect_ratio)
        img = K.eval(tf.image.crop_to_bounding_box(img, 0,(h-w)//2,h,w))
    return img

def main(style,input_file,out_name,original_color,blend_alpha,media_filter):
    img_width = img_height =  256
    #input_file="images/content/"+input_file_name+".jpg"
    out_put="images/generated/"+out_name+"_out.jpg"
    net = nets.image_transform_net(img_width,img_height)
    model = nets.loss_net(net.output,net.input,img_width,img_height,"",0,0)
    model.compile(Adam(),  dummy_loss) 
    model.load_weights("pretrained/"+style+'_weights.h5',by_name=False)
    y = net.predict(x)[0]
    y = crop_image(y)
    ox = crop_image(x[0], aspect_ratio)
    if blend_alpha > 0:
        y = blend(ox,y,blend_alpha)
    if original_color > 0:
        y = original_colors(ox,y,original_color )
    imwrite(out_put, y)
    return out_put
```

In our research, we found that performing some processing after the model generates the image can produce different effects. Here we chose color preservation and style proportion as two effects to process the model-generated images[4](#refer-anchor-4)

python

```python
def original_colors(original, stylized,original_color):#Content image color preservation
    ratio=1. - original_color 

    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)

    hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv_s)    
    return img

def blend(original, stylized, alpha):#Style image proportion
    return alpha * original + (1 - alpha) * stylized
```

#### 3.2.2 Provide API Interface

python

```python
@app.route('/', methods=['POST'])
def index():
    # Receive corresponding parameters
    return_dict= {'return_code': '404',"return_info":"xxx"}
    get_Data=request.form.to_dict()
    style=get_Data["style"]
    original_color=float(get_Data["original_color"])
    blend_alpha=float(get_Data["blend_alpha"])
    media_filter=3
    if request.method == 'POST':
        # Save original image
        uploaded_file = request.files['file']
        print(uploaded_file)
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                path_new,name=generate_filename()
                image_path = os.path.join("images/content", path_new)
                print(image_path)
                uploaded_file.save(image_path)
                print("ready")
                # Call style transfer function
                with graph.as_default():
                    out_path=main(style,image_path,name,original_color,blend_alpha,media_filter)
                # Save generated image
                f = open(out_path, "rb")
                res = f.read()
                s = base64.b64encode(res)
                # Return to cloud server
                return s
    return json.dumps(return_dict)
```

### 3.3 Configure Nginx and uwsgi

When configuring the server, we also used Nginx and uwsgi. Through the flask+Nginx+uwsgi server web deployment, the program can work better during operation and allocate tasks reasonably.

Nginx is good at handling static requests, while uwsgi is good at handling dynamic requests. When we initiate a post request, it is first received and analyzed by Nginx. If it is a dynamic request, Nginx forwards the request to uwsgi for processing through a socket; if it is a static request, Nginx processes it itself and returns the processing result directly to the client, thus basically completing a complete request process.

Here is the Nginx configuration code:

```json
server {
  listen 8080;
  server_name  www.experimentforzcl.cn;

  ssl on;
  ssl_certificate cert/4062906_www.experimentforzcl.cn.pem;
  ssl_certificate_key cert/4062906_www.experimentforzcl.cn.key;
  ssl_session_timeout 5m;
  ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
  ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:HIGH:!aNULL:!MD5:!RC4:!DHE;#°
  ssl_prefer_server_ciphers on;

  location / {
          include uwsgi_params;
          uwsgi_pass 127.0.0.1:6000;
  }
}
```

uwsgi can handle the issue of multiple users requesting simultaneously by processing the program in multiple threads. Here is the configuration code:

```
[uwsgi]
socket = 127.0.0.1:6000
pythonpath = /root/experiment01/fst-2
wsgi-file = /root/experiment01/fst-2/app.py
callable = app
processes = 4
threads = 4
```

## 4. WeChat Mini-Program Introduction

As a photo processing product based on deep learning, creating a good user experience is very important. In today's increasingly popular WeChat, WeChat mini-programs have the characteristic of convenience. With its relatively low development difficulty and widespread use, it has a considerable number of users.

## 5. Reference

<div id="refer-anchor-1"></div>
- [1] [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

<div id="refer-anchor-2"></div>
- [2] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155v1.pdf)

<div id="refer-anchor-3"></div>
- [3] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material]( https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)

<div id="refer-anchor-4"></div>
- [4] [Github](https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py)


```python

```
