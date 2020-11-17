import fasttext

dataset = 'data/fil9'
model = fasttext.train_unsupervised(dataset)

model.save_model("model")

print('model saved')
# model = fasttext.load_model("model")
# x = model.get_word_vector("the")

