Dependancies:
  Python 3
  numpy
  matplotlib
  sklearn
  IPython
  pydotplus

data.zip contains the raw data files:
  processed.cleveland.data
  processed.cleveland.names
  spam.data
  spam.names

data.zip also includes the processed datafiles in the data folder. 
These are binary python pickle dumps of dictionaries that contain the training and test data sets.
The filename_## where the ## represents the proportion of training to test data 

To process data for the other algorithms use:
  python process_data.py <training set %> <input data file> <input name file> <output file name>

The data and name files are csv's where the last column in the data file is the result.
The name file has 2 rows. The first is the names of the variables and the second the names of the results

To run the decsion tree:
  python main.py <file_name> -n <output_filename> -g

-n specifies the filename to save images/graphs with
-g specifies to generate the graphs

To run the random forest:
  python main.py <file_name> -n <out_name>

To run the neural network:
  python main.py <file_name> -n <out_name>