
# Architectural Basics
------------------------------------------------------------------------------------------------------------------------

### In this assignment first we are going to order the 24 topics mentioned in assignment and then we are going to explain the reason behind the order. So let's order them first:

#### 1. Kernels and how do we decide the number of kernels?
#### 2. Convolutions
#### 3. MaxPooling
#### 4. Position of MaxPooling
#### 5. The distance of MaxPooling from Prediction
#### 6. Receptive Field
#### 7. How many layers
#### 8. 1x1 Convolutions
#### 9. Concept of Transition Layers
#### 10. Position of Transition Layer
#### 11. SoftMax
#### 12. DropOut
#### 13. When do we introduce DropOut, or when do we know we have some overfitting
#### 14. Batch Normalization
#### 15. The distance of Batch Normalization from Prediction
#### 16. Image Normalization
#### 17. Learning Rate
#### 18. LR schedule and concept behind it
#### 19. Batch Size, and effects of batch size
#### 20. Number of Epochs and when to increase them
#### 21. When to add validation checks
#### 22. Adam vs SGD
#### 23. How do we know our network is not going well, comparatively, very early
#### 24. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)

### To discuss the reason behind this order we will go through the ordered list one by one. Let's go through it:
* #### For a computer vision problem the very 1st thing we should know about is about kernels or filters. What are they and how to use them for convolution. Then we should know about what should be the no kernels for a given set of image. As we know for a complex dataset we should consider more no of kernels cause there will various complex edges and gradient to learn from image set.
* #### Next Convolution comes into the picture. We should know how actually a 3x3 convolution works. how we can represent a 5x5 convolution using two 3x3 convolution. 
* #### Next we should know about maxpooling as how it works. Also we should learn about how it helps us to tackle slight rotational and positional invariance problem. Also, how its helps to reduce no of layers to an great extent.
* #### After knowing about maxpooling we should concentrate about position of maxpooling as we should use maxpooling atleast when receptive field is 11x11 for a high resolution image. for a low resolution we can use it before that.
* #### Next we should know about position of maxpooling before prediction layer as we don't want to loose any information within last few layer
* #### Next we should have an idea of receptive field like local and global receptive field
* #### After getting idea about receptive field only we can decide no of layers as with increation in resolution in image we need higher global receptive because of which we have increase no of layer in the model.
* #### Next for building convolution layer to should know about 1x1 convolution at it is the most important part of a convolution to combine features and decrease no of channels before maxpooling.
* #### After this we should get some idea about transition layer as it is a combination of 1x1 convolution and a maxpooling layer. Now also we should know for a complicated iamge dataset we are going to define a transition block which we are going to call every time.
* #### Next we should have some idea about the positioning of transition block to properly use it an model as it should be always after convolution layer
* #### Next important thing for building an model is softmax activition as is applied in the output layer. Why should we apply it and what are the drawbacks.
* #### Next and very important model is thing about dropout. like we should know what is the basic idea behind dropout technique. 
* #### Now to use and dropout in a model we should know where to use it and when to use it. As first we should check if there is a huge gap in train and test accuracy then we should go for dropout to reduce overfitting.
* #### Next important thing about a model building is batch normalization as its helps a lot to increase model accuracy by normalizing filters used in the model.
* #### to use an batch normalization layer we should know when to use it and where to use it as we should not use batch normalization at the last layer.
* #### Next thing we should to build an model is image normalization. As we have done batch normalization, the very same way we can use image normalization for better model creation.
* #### After creating a model now we should concentrate on all different compilation technique. Most important of them is learning rate. It actually going to decide how slow or fast model is going to learn.
* #### After learning about learning rate we can use it in a dynamic way as with increase no of epoch we can reduce it so that model can learn steadily.
* #### Next important thing we should concentrate is what should be our batch size. We can try different batch size and choose which one is performing better.
* #### After that to improve model accuracy we should concentrate on what should be out no of epoch. Here also we can try with different value to improve model result.
* #### After learning about epochs we can try to implement validation check in out model so that after each epoch we can see what is the training and test accuracy. And we can consider only that epoch at which it gave maximum validation accuracy.
* #### Next we can concentrate on different optimization technique as ADAM, SGD etc. We can see which one is giving better result based on dataset.
* #### Next we can see the the console after running the model if first two to three epochs does not gives good result then we can come to an conclusion that our model is not performing well and we can try to use some advanced technique
* #### Last but not lease we should check for validation accuracy and if we see after trying all above mention options still model is not performing well then we should increase no or kernel as maybe model is not capturing all texture and gradients or we can try some more advanced techniques.


