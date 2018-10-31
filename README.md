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

<img src="https://github.com/JoeZJH/Labeled-LDA-Python/blob/master/assets/graphical-of-labeled-lda.png" width="400" height="265"/>

* Generative process for Labeled LDA:
<!-- ![https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/generative-process-for-labeled-lda.png](https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/generative-process-for-labeled-lda.png) -->
<img src="https://github.com/JoeZJH/Labeled-LDA-Python/blob/master/assets/generative-process-for-labeled-lda.png" width="400" height="400"/>

* Gibbs sampling equation:
<!-- ![https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/gibbs-sampling-equation.png](https://github.com/JoeZJH/Labeled-LDA/blob/master/assets/gibbs-sampling-equation.png) -->
<img src="https://github.com/JoeZJH/Labeled-LDA-Python/blob/master/assets/gibbs-sampling-equation.png" width="400" height="85"/>

### Usage
* new llda model
* training
* inference
* save model to disk
* load model from disk

##### Example 
```
import model.labeled_lda as llda

# data
labeled_documents = [("example example example example example", ["example"]),
                     ("test llda model test llda model test llda model", ["test", "llda_model"]),
                     ("example test example test example test example test", ["example", "test"])]

# new a Labeled LDA model
llda_model = llda.LldaModel(labeled_documents=labeled_documents)
print llda_model

# training
llda_model.training(iteration=10, log=True)

# inference
document = "test example"
topics = llda_model.inference(document=document, iteration=10, times=10)
print topics

# save to disk
save_model_dir = "../data/model"
llda_model.save_model_to_dir(save_model_dir)

# load from disk
llda_model_new = llda.LldaModel()
llda_model_new.load_model_from_dir(save_model_dir)
print llda_model_new
```


