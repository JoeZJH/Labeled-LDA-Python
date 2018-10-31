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