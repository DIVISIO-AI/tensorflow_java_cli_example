# Tensorflow JAVA CLI example
A simple JAVA command line project loading and executing the SavedModel from the accompanying Jupyter example.

The pre-trained data from the Jupyter notebook is already included. The Application executes predictions for the given SavedModel and CSV data. If the CSV data already contains a column with predicitions, the predictions are compared. This way we can test if our Model still works the same when called from JAVA. (more complex projects should do this in a unit test).

The code itself is under the Apache 2.0 license, the training data used for the model comes with the following note: 


> This dataset is public available for research. The details are described in *Cortez et al., 2009*. 
>  Please include this citation if you plan to use this database:
>
>  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
>  # Modeling wine preferences by data mining from physicochemical properties.
>  ## In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
> 
>  Available at: 
>   * [Elsevier](http://dx.doi.org/10.1016/j.dss.2009.05.016)
>   * [Pre-press (pdf)](http://www3.dsi.uminho.pt/pcortez/winequality09.pdf)
>   * [bib](http://www3.dsi.uminho.pt/pcortez/dss09.bib)

The trained data and the test set are added to the repo, you can override them from the command line (see `RunRegression.java`).
