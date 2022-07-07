# HW4_NLP

First you need to install the requirements:
``pip install -r requirements.txt``

If you want to train the model and to extract the 'Work_For' relation of the input file, you should have in the current working directory the following files:

1. extract.py
2. eval.py
3. model.py

And you need to have a directory named *data* and contained the following files:

1. Corpus.TRAIN.txt
2. TRAIN.annotations
3. Corpus.DEV.txt
4. DEV.annotations

At the end you need to run the following command:
``python extract.py file1 file2``

where:
file1 is the Input file in the .txt format (.processed format are not accepted)
file2 is the Output file in the Annotation format


*To evaluate the performance of the model and compute the f1, recall and precision score*
you need to run the following command:
``python eval.py file1 file2``

where:

file1 is the gold annotation file
file2 is the Output file of the extract program