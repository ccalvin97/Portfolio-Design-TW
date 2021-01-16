# Stock Data Research - GAN & VAE. 



## Keras-GAN, Tensorflow-GAN  
Collection of Keras implementations of Generative Adversarial Networks (GANs) suggested in research papers. These models are in some cases simplified versions of the ones ultimately described in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right.    Contributions and suggestions of GAN varieties to implement are very welcomed.  


What are GANs?  
Generative Adversarial Networks (GANs) are one of the most interesting ideas in computer science today. Two models are trained simultaneously by an adversarial process. A generator ("the artist") learns to create images that look real, while a discriminator ("the art critic") learns to tell real images apart from fakes.  


<b>See also:</b> [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN).  


## Data  
****The testing data are all real TW stock data from web crawling system.****   

**Stock - 福懋(1434)**  
Company Description:  
本公司多角化經營，主要產品有聚胺、聚酯等染整加工織物、純棉織物、混紡織物、長短纖交織布、各種加工機能布、短纖紗支、特殊加工織物、輪胎簾布、PE.塑膠袋、防彈布、阻燃傢飾布、碳纖維與複合材料織物及106個加油站等，已成為世界長纖尼龍、聚酯多富達布產量與品質並優的主要大廠特別在運動領域高品質布材上，最大客戶多年來均為Nike、adidas等世界名牌大企業，與潮流時尚同步發展。  

**Data Example**   
| Time    	| Trading Volume	|Trading Price	|Start Price	|Max Price	|Min Price	|End Price	|Gross Spread	|Trading Count|    
| ---       | ---             |  ---          | ---         |  ---      | ---       |  ---      | ---         |   ---       |  
|109/07/10	|9035900	        |329427750	    |36.95	      |37	        |36.25	    |36.25	    |-0.65      	|        3,207|    
|109/07/09	|8297787	        |306970269	    |37.35	      |37.4	      |36.9	      |36.9	      |-0.45	      |        2,425|    

**Data from 2003 - 2020**     
  
    
## Implementations  

### GAN  
Implementation of _Generative Adversarial Network_ with a MLP generator and discriminator.   
Paper: https://arxiv.org/abs/1406.2661. 

**Conclusion**: It is hard to train the model in order to control both discriminator and generator and Mode Collapse will eastily happen. Please see the picture below.  

**Results**   
<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/Original%20Gan/gan_gereranted_distribution.png" width="420" alt= "tSNE - Visualisation" />
</div>

### WGAN  
Implementation of _Wasserstein GAN_ (with DCGAN generator and discriminator.  
Paper: https://arxiv.org/abs/1701.07875   

**Improvement**  
1. Discriminator do not use Sigmoid in the structure    
2. Loss Function do not use log    
3. Gradient Clipping Design in order to force function be differentiable from limiting the gradient descent  
4. Using Wasserstein Distance Loss  
5. Recommend RMSProp optimiser  


**Conclusion**: It is easier to train the model. However, the gradient vanishing and gradient explosion could happen due to deep neural network. Please see the picture below.   
  
  
**Results**  
***Due to the data is a one-dimensional dataset, the dimension reduction visualisation from tSNE with mean & std check from each feature have been imlemented.***      
| Feature  	|Trading Volume	|Trading Price  |Start Price  |Max Price   |Min Price    |End Price 	 |Gross Spread |Trading Count|
| ---       | ---           |  ---          | ---         |  ---       | ---         |  ---        | ---         |   ---       |
|Real Mean  |0.0400836      |   	0.04548417|	0.542727273	|0.506909548 |	0.544626384|	0.532142857|	0.543922018|	0.457522733|
|Real std   |0.028998539	  |0.031581383	  |0.25354895	  |0.235335259 |	0.256475521|	0.249715523|0.081201313  |0.323748395  |
|Fake Mean  |-0.674068213   |	0.201266319	  |0.370784581  |-0.318771154|-0.646720231 |	0.877627254|	0.319325447|-0.459425181 |
|Fake std   |0.1879538	    |0.538357615	  |0.443408698	|0.445095748 |0.331479222  |	0.161716968|	0.476416647|	0.355708838|

<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/WGAN/wgan_loss.png" width="420" alt= "Loss Plot" />
</div>
<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/WGAN/wgan_distribution_plot.png" width="420" alt= "Distribution between Real and Fake" />
</div>   


### WGAN GP  
Implementation of _Improved Training of Wasserstein GANs_.   
Paper: https://arxiv.org/abs/1704.00028   

**Improvement**  
1. Replace gradient clipping by gradient panelty   

**Conclusion**: It is easier to train the model and easy to be converged. This structure can improve the disadvantage of WGAN. Fisrtly, weight clipping can cause most of the weight on the boundary, which can cause the model cannot generate creative data. Please see the picture below. Secondly, WGAN can easily cause gradient vanishing and explosion due to gradient descent in deep neural network.  

<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/WGAN-GP/weught_clipping.jpg" width="420" alt= "Distribution between Real and Fake" />
</div>     
  
  
  
**Results**  
***Due to the data is a one-dimensional dataset, the dimension reduction visualisation from tSNE with mean & std check from each feature have been imlemented.***      
| Feature  	|Trading Volume	|Trading Price  |Start Price  |Max Price   |Min Price    |End Price 	 |Gross Spread |Trading Count|
| ---       | ---           |  ---          | ---         |  ---       | ---         |  ---        | ---         |   ---       |
|Real Mean  |0.044552693  	|0.056611012  	|0.5175       |	0.482045645|	0.519026753|	0.507087054|	0.561324541|	0.42301741 |
|Real std   |0.050469657    |	0.079521232	  |0.289513054	|0.26893472	 |0.291145955	 |0.284428806	 |0.058725357	 |0.315903139  |
|Fake Mean  |-0.366082102	  |-0.069179036	  |0.190934926	|-0.004797757|	0.021417649|	0.096725188|-0.283989727 |-0.701357067 |
|Fake std   |0.486302137	  |0.297220916	  |0.524810016	|0.422681659 |0.388976485	 |0.438806593	 |0.494849324	 |0.33592689   |

<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/WGAN-GP/wgan-gp_loss.png" width="420" alt= "Loss Plot" />
</div>
<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/WGAN-GP/wgan-gp_distribution_plot.png" width="420" alt= "Distribution between Real and Fake" />
</div>   


### Keras-VAE, Tensorflow-VAE      
 A VAE is a probabilistic take on the autoencoder, a model which takes high dimensional input data compress it into a smaller representation. Unlike a traditional autoencoder, which maps the input onto a latent vector, a VAE maps the input data into the parameters of a probability distribution, such as the mean and variance of a Gaussian. This approach produces a continuous, structured latent space, which is useful for image generation.  
 
Implementation of VAE     
Paper: https://arxiv.org/abs/1312.6114    

**Conclusion**: The performance is bad. It cannot easily generate similar data like other research, which can only generate a vague picture compared to GAN  

**Results**   
<div align="center">
<img src="https://github.com/ccalvin97/fund_design/blob/master/GAN%20and%20VAE/VAE/VAE_distribution_plot.png" width="420" alt= "Visualisation" />
</div>


## Contributing   

Programme is created by Calvin He `<kuancalvin2016@gmail.com>`.  
