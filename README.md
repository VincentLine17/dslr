# dslr

`python3 describe.py dataset.csv`
* Display informations for all numerical features

`python3 histogram.py`
* Answer the following question: Which Hogwarts course has a homogeneous score distribution between the four houses?

`python3 scatter_plot.py`
* Answer the following question: What are the two features that are similar?

`python3 pair_plot.py`
* Display a pair_plot from which we will chose which features are going to use for the logistic regression

`python3 logreg_train.py dataset.csv`
* Train a model from an input dataset generates a file containing the weights that will be used for the prediction

`python3 logreg_predict.py dataset weights`
* Generate a prediction file houses.csv and a visualization file visu_houses.csv than can be used in the scatter_plot script
