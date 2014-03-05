import bz2
import HTMLParser
import os.path
import pdb
import pickle
import pprint
import re
import urlparse
import time
import math

import numpy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.grid_search import GridSearchCV

classifier_fname = "classifier.jblb.pkl"
classifier_path = os.path.join(os.path.dirname(__file__), classifier_fname)

regression_fname = "regression.jblb.pkl"
regression_path = os.path.join(os.path.dirname(__file__), regression_fname)


def htmlclean(s):
  patterns = (
     (re.compile(r'<(script|style|select|comment|option)\b.*?</\1>', re.S | re.I), ''),
     (re.compile(r'<!--.*?-->', re.S), ''),
     (re.compile(r'</?(?:div|p|ul|ol|li|blockquote)\b[^>]*>', re.I), ' '),
     (re.compile(r'<[^>]*>'), ''),
     (re.compile(r'\s+'), ' '),
     )

  for regex, repl in patterns:
    s = regex.sub(repl, s)

  s = HTMLParser.HTMLParser().unescape(s)

  return s.strip()


def transform_item(item):
  if not isinstance(item['content'], list):
    content = item['content']
  else:
    content = item['content'][0]['value']
  content = htmlclean(content)
  length = "xxxdocumentlength%s" % str(int(math.log(len(content)+1)))
  length = "xxxdocumentwordlength%s" % str(int(math.log(len(content)-len(content.replace(' ',''))+1)))
  domain = urlparse.urlparse(item['url']).netloc.replace('.', '_')
  content = ' '.join((
      item['title'],
      content,
      length,
      domain))
  return content
  

def train():
  fname = 'database.pkl.bz2'
  data = pickle.load(bz2.BZ2File(fname, 'r'))

  token_pattern = r'(?u)\b\w[A-Za-z0-9_-]+\w(?:\'s)?\b'

  classifier = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', lowercase=True, token_pattern=token_pattern)),
    ('tfidf', TfidfTransformer(norm='l1')),
    ('clf', svm.SVC(C=100, gamma=1))
  ])

  regressionizer = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=None, lowercase=True, token_pattern=token_pattern)),
    ('tfidf', TfidfTransformer(norm='l1', use_idf=False)),
    ('clf', svm.SVR(gamma=0.01, C=1000.0))
  ])

  def classifier_categorize(item):
    if item['upvotes'] - item['downvotes'] <= 0:
      return 0
    else:
      return 1

  def regression_categorize(item):
    return item['upvotes'] - item['downvotes']



  thing_defs = (
      (classifier, classifier_categorize, classifier_path, None, True, False),
      (regressionizer, regression_categorize, regression_path, mean_absolute_error, False, False))

  purpose = 'generate'

  if purpose=='generate':
    for learner, categorizer, file_path, score, use_gain, output_diag in thing_defs:

      x_train = []
      y_train = []

      for item in data.itervalues():
        content = transform_item(item)
        x_train.append(content)
        category = categorizer(item)
        y_train.append(category)

      y_train = numpy.array(y_train)

      learner.fit(x_train, y_train)

      if output_diag:
        predicted = learner.predict(x_train)
        for item, label, actual in sorted(zip(x_train, predicted, y_train), key=lambda x: x[1]):
          print label, actual, item[:30].encode('utf8')

      scores = cross_validation.cross_val_score(learner, x_train, y_train, cv=5, score_func=score)
      print learner
      print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

      joblib.dump(learner, file_path, compress=9)


  elif purpose=='tune':

    learner, categorizer, file_path, score, use_gain, _ = thing_defs[0]

    x_train = []
    y_train = []

    for item in data.itervalues():
      content = transform_item(item)
      x_train.append(content)
      category = categorizer(item)
      y_train.append(category)

    y_train = numpy.array(y_train)

    parameters = {
      # uncommenting more parameters will give better exploring power but will
      # increase processing time in a combinatorial way
      #'vectorizer__max_df': (0.5, 0.75, 1.0),
      'vectorizer__stop_words': ('english', None),
      #'vectorizer__max_features': (None, 5000, 10000, 50000),
      #'vectorizer__max_n': (1, 2),  # words or bigrams
      #'vectorizer__analyzer': ('word', 'char', 'char_wb'),
      'vectorizer__lowercase': (True, False),
      'vectorizer__token_pattern': (r'(?u)\b\w\w+\b', r'(?u)\b\w\w+(?:\'s)?\b', r'(?u)\b\w+\b', r'(?u)\b[A-Za-z0-9_-]+\b', r'(?u)\b\w[A-Za-z0-9_-]*\w(?:\'s)?\b', r'(?u)\b\w[A-Za-z0-9_-]+\w(?:\'s)?\b', r'(?u)\b\w(?:[A-Za-z0-9_-]+\w)?(?:\'s)?\b'),
      #'tfidf__sublinear_tf': (True, False),
      #'tfidf__use_idf': (True, False),
      #'tfidf__smooth_idf': (True, False),
      #'tfidf__norm': ('l1', 'l2', None),

      # SGDC
      #'clf__alpha': (0.00001, 0.000001),
      #'clf__penalty': ('l2', 'elasticnet'),
      #'clf__n_iter': (10, 50, 80),
      #'clf__alpha': (0.00001, 0.000001),

      # SVM
      #'clf__C': (1, 1e2, 1e3),
      #'clf__gamma': (10, 1, .1, .01, .001),
      #'clf__kernel': ('linear', 'poly', 'rbf'),

      # Bayes
      #'clf__alpha': (0, 1, .1),
      #'clf__fit_prior': (True, False),
    }
     # best score is always the same ?

    eval_kw = 'loss_func' if not use_gain else 'score_func'
    grid_search = GridSearchCV(learner, parameters, n_jobs=-1, verbose=1, cv=4, **{eval_kw: score})

    print "Performing grid search..."
    print "pipeline:", [name for name, _ in learner.steps]
    print "parameters:"
    pprint.pprint(parameters)
    t0 = time.time()
    grid_search.fit(x_train, y_train)
    print "done in %0.3fs" % (time.time() - t0)

    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
      print "\t%s: %r" % (param_name, best_parameters[param_name])

    pprint.pprint(sorted(grid_search.grid_scores_, key=lambda x: x[1], reverse=True))


def test(item):
  s = transform_item(item)

  predicted_category = classifier.predict([s])
  predicted_regression = regression.predict([s])

  return predicted_category[0], predicted_regression[0]

try:
  classifier = joblib.load(classifier_path)
  regression = joblib.load(regression_path)
except Exception:
  pass

if __name__ != '__main__':
  pass

else:
  #print test({'title': 'foo', 'url': 'http://money.com', 'content': ''})
  train()
