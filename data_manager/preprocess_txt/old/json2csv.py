"""
Read in json file dataset_dress_all.json and output csv file with the following columns

img_id,  sent_id,   split,   asin,  folder,   sentence

The csv files are then used to create the matlab structs for karpathy's code.

Before writing to a csv file, the text is cleaned using the methods in pp_utils

"""
import argparse
import json
import csv
from text_clean_utils import get_sentences_from_item


class FoutConfig(object):

    def __init__(self, dir_out, with_zappos):
        if with_zappos == 'no':
            fout = dir_out + 'fashion53k_no_zappos.csv'
        elif with_zappos == 'with_ngrams':
            fout = dir_out + 'fashion53k_with_ngrams.csv'
        elif with_zappos == 'only':
            fout = dir_out + 'fashion53k_only_zappos.csv'
        else:
            raise ValueError("--zappos must be either no, with_ngrams, or only")
        print "file out name", fout
        self.fout = open(fout, 'wb')
        return


def main():

    fieldnames = ['img_id', 'sent_id', 'split', 'asin', 'folder', 'sentence']
    csvfile = FoutConfig(args.data_out_directory, args.zappos).fout

    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()

    with_zappos = args.zappos
    fname_in = args.data_in_fname
    data = json.load(open(fname_in, 'rb'))

    i = 0

    for item in data['items']:
        img_id = item['imgid']
        split = item['split']
        asin = item['asin']
        folder = item ['folder']

        if i % 10000 == 0:
            print "item", i
        sentence_list = get_sentences_from_item(item, with_zappos=with_zappos)

        sent_id = 0
        for l in sentence_list:
            d = {}
            if len(l) == 0:
                continue
            if not l.strip().isspace():
                d['img_id'] = img_id
                d['sent_id'] = sent_id
                d['split'] = split
                d['asin'] = asin
                d['folder'] = folder
                d['sentence'] = l
                writer.writerow(d)

                sent_id += 1

        i += 1

    csvfile.close()

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # specify if you want to use zappos
    parser.add_argument('-z', '--zappos', dest='zappos', type=str, default='only',
                        help='zappos: options: "no", "with_ngrams", "only"')

    parser.add_argument('-din', '--data_in_fname', dest='data_in_fname', type=str,
                        default='../../data/fashion53k/json/dataset_dress_all.json',
                        help='path to json file')

    parser.add_argument('-dout', '--data_out_directory', dest='data_out_directory', type=str,
                        default='../../data/fashion53k/csv/text/',
                        help='where to save the csv file"')

    args = parser.parse_args()

    main()

    # example call from dress_project directory
    #     python data_manager/preprocess_txt/json2csv.py -z no
