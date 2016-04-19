import json
import os


def load_data0(fname='../data0.json'):
    with open(fname, 'r') as f:
        data0 = json.load(f)

    return data0


def write_line2txt(line, file_object):
    for l in line:
        file_object.write(l, ",")
    pass

def remove_space(sentence_set):
    """(set of strings) -> set of strings
    given a set of sentences, remove spaces to compare
    """
    new_set = set()
    for s in sentence_set:
        new_set.add("".join(s.split()))
    return new_set


def verify_nfiles(path1, path2):
    """
    Verifty that the number of files in path1 and path2 are the same
    This is useful for computing dsfit features from a set of images in path 1
    to be stored in path 2
    """

    files1 = [f for f in os.listdir(path1) if not f.startswith(".")]
    files2 = [f for f in os.listdir(path2) if not f.startswith(".")]

    return len(files1) == len(files2)

def verify_nfiles_recursively(path1, path2):

    files1 = [f for f in os.listdir(path1) if not f.startswith(".")]
    files2 = [f for f in os.listdir(path2) if not f.startswith(".")]

    for f in files1:
        p1 = path1 + f
        p2 = path2 + f
        verify_nfiles(p1, p2)
    pass


def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    """
    method may be used to save a numpy array compactly.
    I used for saving the cnn matrices
    http://stackoverflow.com/questions/24691755/how-to-format-in-numpy-savetxt-such-that-zeros-are-saved-only-as-0
    """
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')


if __name__ == '__main__':

    # see populate_excluded_phrases_user_input.py
    pass


