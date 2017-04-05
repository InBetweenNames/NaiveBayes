# Multinomial Naive Bayes engine

Author: Shane Peelar (peelar@uwindsor.ca)

If you choose to use this tool in your project, please cite me in your report.  You can use this BiBTeX entry to conveniently do so:

~~~
@article{peelar, title={Multinomial Naive Bayes Engine}, url={https://github.com/InBetweenNames/NaiveBayes}, author={Peelar, Shane M}} 
~~~

This code is intended for the 60-538 course Information Retrieval at the University of Windsor.  A Visual Studio 2017 solution is provided for the code.

~~~
	Usage: ./MultinomialNaiveBayes (--selectfeatures (<n>)) [class1 class1filename class2 class2filename ... classN classNfilename]
	If no arguments are provided, this command line will be run: ./MultinomialNaiveBayes--selectfeatures 100 icse metadata/icse_id.txt vldb metadata/vldb_id.txt
	If --selectfeatures is provided, then an M-fold cross validation test will be run to determine the optimal feature size
	If --selectfeatures <N> is provided, a classifier will be built with the N best features and written out to classifier.dat
~~~

Although arbitrary M-fold cross validation is supported, currently M is hardcoded to 10 in the code.  If it is desireable to change this,
just send me an email and I can add a command line argument.

The file `classifier.dat` that is produced with the `--selectfeatures <n>` command has all information required for an online classifier to work (it is the training model).
An example file is provided.  You can use this to perform training offline, and export your results to an online classifier.  It also means you don't need to burden
your webserver with performing training.
