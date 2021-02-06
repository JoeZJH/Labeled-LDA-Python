import sys
sys.path.append('../')
import model.labeled_lda as llda

# initialize data
labeled_documents = [("example example example example example"*10, ["example"]),
                     ("test llda model test llda model test llda model"*10, ["test", "llda_model"]),
                     ("example test example test example test example test"*10, ["example", "test"]),
                     ("good perfect good good perfect good good perfect good "*10, ["positive"]),
                     ("bad bad down down bad bad down"*10, ["negative"])]

# new a Labeled LDA model
# llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector="50_div_K", eta_vector=0.001)
# llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector=0.02, eta_vector=0.002)
llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector=0.01)
print(llda_model)

# training
# llda_model.training(iteration=10, log=True)
while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))
    llda_model.training(1)
    print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
    print("delta beta: %s" % llda_model.delta_beta)
    if llda_model.is_convergent(method="beta", delta=0.01):
        break

# update
print("before updating: ", llda_model)
update_labeled_documents = [("new example test example test example test example test", ["example", "test"])]
llda_model.update(labeled_documents=update_labeled_documents)
print("after updating: ", llda_model)

# train again
# llda_model.training(iteration=10, log=True)
while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))
    llda_model.training(1)
    print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
    print("delta beta: %s" % llda_model.delta_beta)
    if llda_model.is_convergent(method="beta", delta=0.01):
        break

# inference
# note: the result topics may be different for difference training, because gibbs sampling is a random algorithm
document = "example llda model example example good perfect good perfect good perfect" * 100

topics = llda_model.inference(document=document, iteration=100, times=10)
print(topics)

# perplexity
# calculate perplexity on test data
perplexity = llda_model.perplexity(documents=["example example example example example",
                                              "test llda model test llda model test llda model",
                                              "example test example test example test example test",
                                              "good perfect good good perfect good good perfect good",
                                              "bad bad down down bad bad down"],
                                   iteration=30,
                                   times=10)
print("perplexity on test data: %s" % perplexity)
# calculate perplexity on training data
print("perplexity on training data: %s" % llda_model.perplexity())

# save to disk
save_model_dir = "../data/model"
# llda_model.save_model_to_dir(save_model_dir, save_derivative_properties=True)
llda_model.save_model_to_dir(save_model_dir)

# load from disk
llda_model_new = llda.LldaModel()
llda_model_new.load_model_from_dir(save_model_dir, load_derivative_properties=False)
print("llda_model_new", llda_model_new)
print("llda_model", llda_model)
print("Top-5 terms of topic 'negative': ", llda_model.top_terms_of_topic("negative", 5, False))
print("Doc-Topic Matrix: \n", llda_model.theta)
print("Topic-Term Matrix: \n", llda_model.beta)
