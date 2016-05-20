About this directory
==============

This directory contains roughly two groups of files:
  
  - Preprocessing of text to be used with the **LSTM model from tensorflow**.
  - Preprocessing of text to be used to generate the **matlab structures for Karpathy**.
  

For LSTM processing
--------------

1) Create interim files:

    $ python concat_all_txt_interim.py -s train -z no 

s stands for split and you can choose: train, val, test

z stands for zappos and you can choose: no, with_ngrams, only.

2) Process interim files to create the text files to be fed to the LSTM from tensorflow:

    $ python create_text_data.py

You can run this script by setting the with_zappos to either:  no_zappos, with_ngrams, only_zappos

This script reads the interim files and adds <sos>, <eos>, <unk>, for start of sentence, end of sentence and unkown token according to a vocabulary. The vocabulary is extracted from th interim files and we keep words that happen more than 5 times


For creating matlab structs for Karpathy
--------------

1) Create csv files from json:

    $ python data_manager/preprocess_txt/json2csv.py -z no

z stands for zappos and you can choose: no, with_ngrams, only.

2) Save the vocabulary file
    
    get_vocab_from_scv.py

2) Create text matlab structs for Karpathy from csv files

    create_txt_struct_4karpathy
    
    

*This will be Italic*

**This will be Bold**

- This will be a list item
- This will be a list item

    Add a indent and this will end up as code