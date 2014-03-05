Rudimentary attempt at upvote prediction

learn/grab_data.py downloads and archives link/vote data

learn/train.py learns models based on archived link/vote data

reddit_sync.py reads unseen links from an rss feed and rates them based on the models and submits them if they pass thresholds.
