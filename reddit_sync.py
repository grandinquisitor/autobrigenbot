import cookielib
from collections import defaultdict
import pprint
import re
import sys
import time
import urllib
import urllib2
import warnings

import feedparser
import simplejson

from learn.train import test as classify

# https://github.com/reddit/reddit/wiki/API: "make no more than one request per two seconds
rate_limit = cfg.rate_limit
post_limit = cfg.post_limit

percent_filter = cfg.percent_limit

subreddit = cfg.subreddit
user = cfg.subreddit
passwd = cfg.password
feed_loc = cfg.feed_loc  # expects input aggregated into a single rss stream
user_agent = cfg.user_agent

banned_sites = cfg.banned_sites
banned_titles = cfg.banned_titles

interactive = '-i' in sys.argv

feed = feedparser.parse(feed_loc)

title_hasher = lambda x: re.compile(r'\W+').sub('', x.lower()) if x else ''

cj = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
opener.addheaders = [('User-agent', user_agent)]

existing_stuff = simplejson.load(opener.open('http://www.reddit.com/r/%s.json' % subreddit))
existing_titles = set(title_hasher(i['data']['title']) for i in existing_stuff['data']['children'])
existing_urls = set(i['data']['url'] for i in existing_stuff['data']['children'])

time.sleep(rate_limit)


r = opener.open('http://www.reddit.com/api/login/%s/' % user, data=urllib.urlencode({'user': user, 'passwd': passwd, 'api_type': 'json'}))

json = simplejson.load(r)
modhash = json['json']['data']['modhash']
time.sleep(rate_limit)

story_queue = []

for entry in feed.entries:
	url, title = entry.link, entry.title

	if title_hasher(title) in existing_titles:
		print "skipping title match %r" % title
		continue

	if entry.link in existing_urls or (entry.id and entry.id in existing_urls):
		print "skipping url match %r" % entry.link
		continue

	# feedburner correction
	# would be nice if we could get planet to do this automatically
	if 'feedproxy.google' in url:
		if re.compile(r'^http://\w+').match(entry.id):
			print "attempting feedburner correction", url, entry.id
			url = entry.id

	# have seen it where there is no url but the id is good
	# e.g., rationally speaking
	elif not url and re.compile(r'http://\w+').match(entry.id):
		url = entry.id

	if not set(filter(None, (entry.link, entry.id, url))).isdisjoint(existing_urls):
		print "skipping url match %r" % entry.link
		continue

	kill_it_now = False
	for domain in banned_sites:
		if re.compile(r'^\w+:/*[^/]*\b' + re.escape(domain) + r'(?:/|$)', re.I).match(url):
			print "%s (%s) is known banned site, skipping" % (url, domain)
			kill_it_now = True
			break

	for banned_title in banned_titles:
		if banned_title in title:
			kill_it_now = True
			break
	
	if kill_it_now:
		continue

	r = opener.open('http://www.reddit.com/api/info.json?url=%s' % urllib.quote_plus(url))
	json = simplejson.load(r)

	subreddits = frozenset(submission['data']['subreddit'] for submission in json['data']['children'])
	subreddits = ()

	if subreddit not in subreddits:
		story_queue.append((url, title, entry))
		existing_titles.add(title_hasher(title))
		existing_urls.add(url)

	time.sleep(rate_limit)


# pass it through both regression and classification
# classifier just predicts score < 1, and skips
# otherwise, post just the top scoring result

ok_queue = []
for url, title, entry in story_queue:
	category, score=classify(
			{'title': title,
			'url': url,
			'content': entry.get('content', entry.summary)})

	if category == 1:
		ok_queue.append((url, title, score))
	else:
		print "filtering", url

ok_queue.sort(key=lambda x: x[2], reverse=True)
pprint.pprint(ok_queue)

attempts = 0
for url, title, _ in ok_queue:
	# reddit won't accept links w/o title
	if not title:
		continue

	try:
		data = {
			'url': url.encode('utf8'),
			'title': title.encode('utf8'),
			'sr': subreddit,
			'kind': 'link',
			'uh': modhash,
			'api_type': 'json',
		}
		r = opener.open('http://www.reddit.com/api/submit', data=urllib.urlencode(data))
		json = simplejson.load(r)

		if 'captcha' in json['json']:
			if not interactive:
				raise RuntimeError("need captcha")

			captcha_id = json['json']['captcha']
			captcha_url = "http://www.reddit.com/captcha/%s.png" % captcha_id
			print captcha_url
			solution = raw_input("--> ").strip()
			data['captcha'] = solution
			data['iden'] = captcha_id

			continue
			r = opener.open('http://www.reddit.com/api/submit', data=urllib.urlencode(data))

		elif 'errors' in json['json'] and json['json']['errors']:
			if any('DOMAIN_BANNED' in err for err in json['json']['errors']):
				warnings.warn(repr(json), RuntimeWarning)
			elif any('ALREADY_SUB' in err for err in json['json']['errors']):
				warnings.warn(repr(json), RuntimeWarning)
			else:
				raise RuntimeError(json)

		print 'posted link %r %r' % (url, title)
	except Exception:
		print "got fail"
	else:
		attempts += 1
	finally:
		if attempts >= post_limit:
			break
		else:
			time.sleep(rate_limit)
