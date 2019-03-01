import model.labeled_lda as llda

# data
labeled_documents = [("example example example example example", ["example"]),
                     ("test llda model test llda model test llda model", ["test", "llda_model"]),
                     ("example test example test example test example test", ["example", "test"]),
                     ("good perfect good good perfect good good perfect good ", ["positive"]),
                     ("bad bad down down bad bad down", ["negative"])]

# new a Labeled LDA model
# llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector="50_div_K", eta_vector=0.001)
# llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector=0.02, eta_vector=0.002)
llda_model = llda.LldaModel(labeled_documents=labeled_documents)
print llda_model

# training
# llda_model.training(iteration=10, log=True)
while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))
    llda_model.training(1)
    print "after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity)
    if llda_model.is_convergent:
        break

# update
print "before updating: ", llda_model
update_labeled_documents = [("new example test example test example test example test", ["example", "test"])]
llda_model.update(labeled_documents=update_labeled_documents)
print "after updating: ", llda_model

# train again
# llda_model.training(iteration=10, log=True)
while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))
    llda_model.training(1)
    print "after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity)
    if llda_model.is_convergent:
        break

# inference
# note: the result topics may be different for difference training, because gibbs sampling is a random algorithm
document = "example llda model example example good perfect good perfect good perfect"
# topics = llda_model.inference(document=document, iteration=10, times=10)
topics = llda_model.inference_multi_processors(document=document, iteration=10, times=10)
print topics

# save to disk
save_model_dir = "../data/model"
llda_model.save_model_to_dir(save_model_dir)

# load from disk
llda_model_new = llda.LldaModel()
llda_model_new.load_model_from_dir(save_model_dir)
print "llda_model_new", llda_model_new
print "llda_model", llda_model
