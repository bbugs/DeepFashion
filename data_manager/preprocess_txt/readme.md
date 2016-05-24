About this directory
==============

This directory contains roughly three groups of files:
--------------

  - Read the original json file and write a new one with clean text
  - Preprocessing of text to be used with the **LSTM model from tensorflow**.
  - Preprocessing of text to be used to generate the **matlab structures for Karpathy**.
  


Cleaning the text on the original json files
--------------

- Read the original json file and write a new one with clean text.  There is an option to consider an external vocabulary (e.g., zappos).

From the mac, while Paris directories are mounted, you can run
    
    $ python data_manager/preprocess_txt/data_cleaner.py

*data_cleaner.py* requires *text_clean_utils.py*

You should run it from the mac because it has the nltk file.  In the future, download the nltk data in Paris.



To Create files for LSTM processing
--------------
    $ python preprocess_txt/data_preprocessory.py

Simple change the split and level of zappos vocab in the main section.  Note the ptb model reader later adds <eos> tokens.



For creating matlab structs for Karpathy
--------------

1) Create csv files from json:

    $ python data_manager/preprocess_txt/json2csv.py -z no

z stands for zappos and you can choose: no, with_ngrams, only.

2) Save the vocabulary file
    
    get_vocab_from_scv.py

2) Create text matlab structs for Karpathy from csv files

    create_txt_struct_4karpathy
    
