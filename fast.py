import fasttext

dataset = 'data/fil9'
# model = fasttext.train_unsupervised('data/corpus/R8.clean.txt', dim=300)
model = fasttext.train_unsupervised(dataset, dim=400)

model.save_model("model")

print('model saved')
# model = fasttext.load_model("model")
# x = model.get_word_vector("the")

