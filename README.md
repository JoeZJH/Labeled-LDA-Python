## Implement of L-LDA Model(Labeled Latent Dirichlet Allocation Model) with python


References:
   * *Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora, Daniel Ramage...*
   * *Parameter estimation for text analysis, Gregor Heinrich.*
   * *Latent Dirichlet Allocation, David M. Blei, Andrew Y. Ng...*
   
### An efficient implementation based on Gibbs sampling

**The following descriptions come from *Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora, Daniel Ramage...***

##### Introduction:
Labeled LDA is a topic model that constrains Latent Dirichlet Allocation by defining a one-to-one correspondence between LDA’s latent topics and user tags.
Labeled LDA can directly learn topics(tags) correspondences.

##### Gibbs sampling:
* Graphical model of Labeled LDA:
<!-- ![https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/graphical-of-labeled-lda.png](https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/graphical-of-labeled-lda.png) -->

<img src="https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/graphical-of-labeled-lda.png" width="400" height="265"/>

* Generative process for Labeled LDA:
<!-- ![https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/generative-process-for-labeled-lda.png](https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/generative-process-for-labeled-lda.png) -->
<img src="https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/generative-process-for-labeled-lda.png" width="400" height="400"/>

* Gibbs sampling equation:
<!-- ![https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/gibbs-sampling-equation.png](https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/gibbs-sampling-equation.png) -->
<img src="https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/gibbs-sampling-equation.png" width="400" height="85"/>




