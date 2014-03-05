import bz2
import HTMLParser
import pickle
import pprint
import sys
import urlparse

import requests
import feedparser
import yaml

import feedfinder

sys.path.append('..')
from cfg import cfg

def iterate():
  baseurl = 'http://www.reddit.com/r/%s/.json?count=100' % cfg.subreddit
  after = None
  for _ in xrange(2):
    result = requests.get(baseurl + ('&after=' + after if after else ''))
    try:
      json = result.json()
    except TypeError:
      json = result.json

    for thing in json['data']['children']:
      print thing['data']['url']
      yield thing['data']

    after = json['data']['after']


_feed_cache = {}

planetrationalist_feed = feedparser.parse(cfg.aggregate_feed_url)

def find_entries(already_found):
  for data in iterate():
    url = data['url']
    url = url.replace('&amp;', '&')
    found_entry = None

    if url in already_found:
      already_found[url]['upvotes'] = data['ups']
      already_found[url]['downvotes'] = data['downs']
      continue

    if not found_entry:
      try:
        feed_urls = list(feedfinder.extract_feed_links(requests.get(url).text))
        for feed_url in feed_urls:
          feed_url = urlparse.urljoin(url, feed_url)
          _feed_cache[feed_url] = feed = _feed_cache.get(feed_url, feedparser.parse(feed_url))
          for entry in feed.entries:
            if url in (entry.get('id'), entry.get('link'), entry.get('feedburner_origlink')):
              print "found in original rss"
              found_entry = entry
              break

          if found_entry:
            break
      except Exception, e:
        print e
        pass

    if not found_entry:
      for entry in planetrationalist_feed.entries:
        if url in (entry.link, entry.id):
          print "found in cached rss"
          found_entry = entry
          break

    if found_entry:
      yield {'upvotes': data['ups'], 
          'downvotes': data['downs'],
          'url': url,
          'title': found_entry['title'],
          'content': found_entry.get('content', found_entry.get('summary', '')),
          'timestamp': data['created']}


fname = 'database.pkl.bz2'
try:
  data = pickle.load(bz2.BZ2File(fname, 'r')) or {} 
except IOError:
  data = {}

try:
  for row in find_entries(data):
    if all(v != '' and v is not None for v in row.itervalues()):
      data[row['url']] = row

finally:
  if data:
    pickle.dump(data, bz2.BZ2File(fname, 'w'))
    print len(data)
