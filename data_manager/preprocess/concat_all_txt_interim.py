
import argparse
import json

from pp_utils import get_sentences_from_item


class FoutConfig(object):
    def __init__(self, dir_out, with_zappos, split):
        if with_zappos == 'no':
            fout = dir_out + '/no_zappos/fashion53k.{}.txt'.format(split)
        elif with_zappos == 'with_ngrams':
            fout = dir_out + '/with_ngrams/fashion53k.{}.txt'.format(split)
        elif with_zappos == 'only':
            fout = dir_out + '/only_zappos/fashion53k.{}.txt'.format(split)
        else:
            raise ValueError("--zappos must be either no, with_ngrams, or only")
        print "file out name", fout
        self.fout = open(fout, 'w')
        return


def main(params):

    with_zappos = params['zappos']
    fname_in = params['data_in_fname']
    data = json.load(open(fname_in, 'rb'))
    split = params['split']
    dir_out = params['data_out_directory']

    # clean and concatenate the text from all the items
    i = 0
    text = ''
    file_out_handle = FoutConfig(dir_out, with_zappos, split).fout
    print "processing json file"
    for item in data['items']:
        if i % 10000 == 0:
            print "item", i
        if item['split'] == split:
            sentence_list = get_sentences_from_item(item, with_zappos=with_zappos)
            for l in sentence_list:
                if len(l) == 0:
                    continue
                if not l.strip().isspace():
                    file_out_handle.write(l + '\n')
            # text  += t
        i += 1

    file_out_handle.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--split', dest='split', type=str,
                        default='train',
                        help='split options: "train", "val", "test"')

    # specify if you want to use zappos
    parser.add_argument('-z', '--zappos', dest='zappos', type=str, default='only',
                        help='zappos: options: "no", "with_ngrams", "only"')

    parser.add_argument('--data_in_fname', dest='data_in_fname', type=str,
                        default='../../../../data/fashion53k/json/dataset_dress_all.json',
                        help='path to json file')

    parser.add_argument('--data_out_directory', dest='data_out_directory', type=str,
                        default='../../../../data/fashion53k/text/concat_interim/',
                        help='zappos: options: "no", "with_ngrams", "only"')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed parameters:'
    print json.dumps(params, indent=2)
    main(params)

    # example call
    # python concat_all_txt_interim.py -s train -z no
