### This is pretty trivial to implement

### The hard part is deciding who you want to recognize, and then collecting a ton of photos.

save them to the dataset directory. 

process.py will take the whole dataset and quantify every face with 128-D encodings,
and save them with their associated names.
it uses a pre-trained model to find faces in a photo, and another pre-trained model that quanities a face into the embedding.

tran_model.py will take those embeddings and names, and do the machine learning.
that means learning to recognize who each face belongs to.

a default SVC or MLP model is maybe good enough, but I've tried tweaking some custom built models to get better accuracy. 
but in any case, better data - more photos, and better quality photos - is the key to better accuracy.

classify.py is also pretty trivial. just upload a photo. it will find any faces in it,
and compare them with the faces that the model has been trained to recognize.

in any case, again, the model is only as good as the data it has been trained with. 
