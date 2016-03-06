__author__ = 'david_torrejon'

import sys


from file_preparation import dataset_preparation
from analogies_finder import find_analogies


def main(argv):
    print ('TBIR 15-16 Assignemnt')
        #print argv[0]
    try:
        prepared_ds_fn, solution_fn = dataset_preparation(argv[0])
    except:
        print('You gave no route for the file')
        prepared_ds_fn, solution_fn =  dataset_preparation()

    print('datasets ready...')
    find_analogies(prepared_ds_fn, solution_fn)

if __name__ == "__main__":
    main(sys.argv[1:]) # all except the name of the execution
