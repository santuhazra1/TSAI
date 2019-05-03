
## Assignment 1B

#### What are Channels and Kernels?
Filter is used in Convolutional neural network to extarct different features of an image. there are different filter used to extract different features of an image as  horizntal filter, vertical filter, Edge detector, gray scale filter etc. Tipically in a convolution network filter weights are initialized as random and later later through  backpropagation network update its weight to extract better feature. Genarally if an nxn image is convolve with an fxf filter with padding p and stride s then its output dimension is going to be (n+2p-f+1/s,n+2p f+1/s).
![alt text](https://cdn-images-1.medium.com/max/1600/1*7S266Kq-UCExS25iX_I_AQ.png)


#### Why should we only (well mostly) use 3x3 Kernels?
Number of parameter used for 3x3 is 9. So to get an output equivalent to 5x5 we have to use two layes of 3x3 filter. So, total no of parameter used for two layers of 3x3 filter is 18. Now for an equivalent 5x5 filter total number parameter is 25. So, by using 3x3 filter no of parameter reduces which saves memory. To get an output equivalent to higher than 5x5 we just have to use more no of 3x3 layer.

#### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199?

To reach 1x1 from 199x199 we have to use 99 times 3x3 convolution as after applying 99 times 3x3 convolution the global receptive field will become 199x199.
Below are the steps for reaching 199x199  to 1x1 by using 3x3 convolution:

199| 197| 195| 193| 191| 189| 187| 185| 183| 181| 179| 177| 175| 173| 171| 169| 167| 165| 163| 161| 159| 157| 155| 153| 151| 149| 147| 145| 143| 141| 139| 137| 135| 133| 131| 129| 127| 125| 123| 121| 119| 117| 115| 113| 111| 109| 107| 105| 103| 101| 99| 97| 95| 93| 91| 89| 87| 85| 83| 81| 79| 77| 75| 73| 71| 69| 67| 65| 63| 61| 59| 57| 55| 53| 51| 49| 47| 45| 43| 41| 39| 37| 35| 33| 31| 29| 27| 25| 23| 21| 19| 17| 15| 13| 11| 9| 7| 5| 3| 1
