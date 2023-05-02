Download Link: https://assignmentchef.com/product/solved-comp135-project-2
<br>
For this assignment, you will explore a variety of classifiers, applied to a real-world data-set for natural language processing. In each case, you will explore and analyze a range of model hyperparameters.

<h1>Data for the Project</h1>

You have been supplied with several thousand single-sentence reviews, collected from three domains: imdb.com, amazon.com, and yelp.com. Each review consists of a sentence, and has been assigned a binary label indicating the <em>sentiment </em>(1 for positive and 0 for negative) of that sentence. Your goal is to develop binary classifiers that can generate the sentiment-labels for new sentences, automating the assessment process. While the reviews were collected from websites where much of the content is in English, the reviews may well contain slang, spelling errors, foreign characters and the like, all of which make natural language data challenging, albeit fun, to try to classify like this.

The provided data consists of 2<em>,</em>400 training examples in the usual CSV x and y format.<sup>∗ </sup>Input data has two columns, for the source-website and review text; outputs are given as binary values, where 1 indicates a positive review. There are also 600 testing inputs, for which no y-values are given; these will be used for validation against the Gradescope leaderboards. The Project download also contains a short script, load_train_data.py, that will give you guidance as to how you might load the data using Pandas (of course, you can load it in other ways as well if you so choose).

Examples of <em>positive </em>reviews include:

<ul>

 <li>(amazon) #1 It Works – #2 It is Comfortable.</li>

 <li>(imdb) “Gotta love those close-ups of slimy, drooling teeth! “</li>

 <li>(yelp) Food was so gooodd.</li>

</ul>

Examples of <em>negative </em>reviews include:

<ul>

 <li>(amazon) DO NOT BUY DO NOT BUYIT SUCKS</li>

 <li>(imdb) This is not movie-making.</li>

 <li>(yelp) The service was poor and thats being nice.</li>

</ul>

∗

Data comes from work by D. Kotzias, M. Denil, N. De Freitas, and P. Smyth, as described in their paper <a href="http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf">From </a><a href="http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf">Group to Individualized Labels Using Deep Features</a> (KDD 2015). Thanks to the authors for making it available.

It is recommended that you <em>preprocess </em>your data, removing punctuation, non-English and nontext characters, and unifying the case (i.e., setting everything to be either upper- or lower-case). You will then investigate <em>feature representations </em>for converting strings of words in sentence form into feature vectors <strong>x</strong><em><sub>n </sub></em>of some common length <em>n</em>, and use those feature representations to build and compare a number of different types of models.

<h1>Part Zero: Collaborators file</h1>

Provide the usual file containing your name, the amount of time you worked on the assignment, and any resources or individuals you consulted in your work. Note that for this assignment, there are <em>no restrictions </em>on the Python libraries you may use (you won’t be handing in code), along with the usual sklearn models and functions; you can consult other sites and guides to those libraries, but should include any such resources in your collaborators file as well.

<h1>Part One: Classifying Review Sentiment with Bag-of-Words Features</h1>

The “Bag-of-Words” (BoW) model of a document (i.e., in this case, a single review) involves determining a known fixed vocabulary, <em>V </em>, in advance, imposing an order on those words, and then representing each document with a vector of length |<em>V </em>| that has a non-zero value at position <em>i </em>if the <em>i</em>th word in <em>V </em>is part of that document, and is 0 otherwise.<sup>† </sup>You will build such a representation for your input data (train and test). Your first step will be to make some design decisions with respect to how your BoW model works; questions you will need to answer may include:

<ul>

 <li>How big is the vocabulary, and what order to you place those words into?</li>

 <li>Do you exclude very rare words (and what does “very rare” mean)?</li>

 <li>Do you exclude very common words (and what does “very common” mean)?</li>

 <li>Do you count the occurrences of a word in the document, or only record if it is there or not (producing a binary vector)?</li>

 <li>Is it worth using something other than word counts, like the <em>inverse document frequency </em>idea described in lecture.</li>

 <li>Do you use single features only, or do you try counting word-pairs instead? What about counting <em>n</em>-tuples of words?</li>

</ul>

Whatever you decide (and you may want to experiment) you want a representation whereby each feature of the resulting input vector corresponds to a single word (or <em>n</em>-tuples of words, if you go that route). Once you have decided upon your feature representation, you will investigate three distinct classifier models on the data, seeking one that gives you best performance.

<strong>Resources</strong>: there are several tools available in sklearn for creating BoW representations:

<a href="https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation">https://scikit-learn.org/stable/modules/feature</a> <a href="https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation">extraction.html</a>

<ol>

 <li>(<em>10 pts.</em>) In your report, include a paragraph or two that explain the “pipeline” for generating your BoW features. This should include a clear description of any pre-processing you did on the basic text, along with the sorts of decisions you made in generating your final featurevectors. You should present this in complete enough form that someone else (another student,</li>

</ol>

†

You can find some discussion of this in the material on clustering, from the end of the course, which has already been released.

say) could produce a model identical to yours if they wished, based upon reading your report. As we have said before, keep code samples to a minimum; ideally, you should be able to explain what you did in plain language. Your paragraph should also contain some justification for why you made the decisions you did.

<ol start="2">

 <li>Generate a <em>logistic regression </em>model for your feature-data and use it to classify the training data. In your report:

  <ul>

   <li>Give a few sentences describing the model you built, and any decision made about how you set is parameters, trained it, etc.</li>

   <li>Choose at least two hyperparameters that control model complexity and/or its tendency to overfit. Vary those hyperparameters in a systematic way, testing it using a crossvalidation methodology (you can use libraries that search through and cross-validate different hyperparameters here if you like). Explain the hyperparameters you chose, the range of values you explored (and why), and describe the cross-validation testing in a clear enough manner that the reader could reproduce its basic form, if desired.</li>

   <li>Produce at least one figure that shows, for at least two tested hyperparameters, performance for at least 5 distinct values—this performance should be plotted in terms of average error for both training and validation data across the multiple folds, for each of the values of the hyperparameter. Include information, either in the figure, or along with it in the report, on the <em>uncertainty </em>in these results.<sup>‡</sup></li>

   <li>Give a few sentences analyzing these results. Are there hyperparameter settings for which the classifier clearly does better (or worse)? Is there evidence of over-fitting at some settings?</li>

  </ul></li>

 <li>Generate a <em>neural network </em>(or MLP) model for you feature-data. Produce the same sort of description and analysis for it as you did for the previous model, including variation of two or more hyperparameters, cross-validation testing, and at least one figure for each hyperparameter (minimum two) that shows how performance on training and validation data is affected as the hyperparameters change.</li>

 <li>Generate a third model, of whatever type you choose; you could use, for instance, SVM classifiers, or try ones that we have not yet explored directly (sklearn has its own decision-tree and decision-forest classifiers, for example). Whatever you choose, produce the same analysis as for the prior models, including a description of what you did, how hyperparameter variation affected results, and so forth. Figures are expected showing training/validation performance relative to hyperparameter variation; additional figures are allowed, of course.</li>

 <li>Summarize which classifier of the three you built performs best overall on your labeled data, and give some reasons why this may be so. Does it have more flexibility? Is it better at avoiding overfitting on this data?</li>

</ol>

In addition, look at the performance of your best classifier and try to characterize the mistakes that it makes. Are there common features to the sentences that it gets wrong (e.g., are they

‡

This can be measured in terms of simply standard deviation across the <em>k</em>-fold cross-validation tests, or in more detail by showing exact performance metrics on each fold. The idea is to help the reader understand if the average performance is typical and stable, or if there is a lot of difference from one cross-validation test to another.

mostly from one of the three source websites)? Are there other features that you can identify? Can you hypothesize why you see the results you do?

<ol start="6">

 <li>Apply your best classifier from the previous steps to the text data in csv file, storing the outcomes as a probabilistic prediction and then submitting them to the leaderboard, as described below. In your report, describe the performance that you see there. How does that match up to the performance you saw during training and cross-validation? If it is as expected, what does that tell us, do you think? If it is not as expected, what does <em>that </em>tell us?</li>

</ol>

<h1>Part Two: Prediction submissions</h1>

To test your various classifiers, you can submit the predictions that each makes—on the unlabeled x_test.csv file—to a leaderboard. You can submit output from multiple classifiers, of multiple types, and simply re-submit whichever did the best at the end for your final graded score. The leaderboard code will compare your predictions to known correct examples, scoring them relative to the correct answers.

As for Project 01, the submission should be in the form of a plain text-file, named yproba1_test.txt, containing one probability value (a floating-point number giving the probability of a positive binary label, 1) per example in the test input. Each line will be a single number, and we should be able to load it into a 1-dimensional NumPy array using:

np.loadtxt(‘yproba1_test.txt’)

(It would be a good idea to verify that this will work as expected.) These numbers will be thresholded at a probability of 0<em>.</em>5 for scoring purposes.