# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:18:47 2023

@author: cianw
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re


redditComments = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\redditCommentsFoodInflation.csv")
redditComments['Date']= pd.to_datetime(redditComments['Date'],unit='s')
redditComments = redditComments[redditComments['Comment']!='[deleted]']
redditComments = redditComments[redditComments['Comment']!='[removed]']
redditComments = redditComments[redditComments['Comment']!='']

filter_char = lambda c: ord(c) < 256
redditComments['Comment_adj'] = redditComments['Comment'].apply(lambda s: ''.join(filter(filter_char, s)))
redditComments['Comment_adj'] = redditComments['Comment_adj'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
redditComments['Comment_adj'] = redditComments['Comment_adj'].str.casefold()
redditComments['Comment_adj'] = redditComments['Comment_adj'].str.replace('[^a-zA-Z ]', '')

stop_words = nltk.corpus.stopwords.words("english")
redditComments['Comment_adj'] = redditComments['Comment_adj'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
redditComments['Comment_adj'] = redditComments['Comment_adj'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>1]))

all_words = ' '.join([word for word in redditComments['Comment_adj']])
tokenized_words = nltk.tokenize.word_tokenize(all_words)
fdist = FreqDist(tokenized_words)
fdist

#redditComments['Comment_adj'] = redditComments['Comment_adj'].apply(lambda x: ' '.join([item for item in x.split() if fdist[item] == 1 ]))
wnl = WordNetLemmatizer()
redditComments['Comment_lem'] = redditComments['Comment_adj'].apply(wnl.lemmatize)
redditComments['is_equal']= (redditComments['Comment_adj']==redditComments['Comment_lem'])
# show level count
redditComments.is_equal.value_counts()

allWordsLem = ' '.join([word for word in redditComments['Comment_lem']])

"""
******************************************************************************************************************************
Word Cloud?
******************************************************************************************************************************
"""

plt.figure(figsize=(10, 7))
ax1 = wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(allWordsLem)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');
plt.show()


x, y = np.ogrid[:400, :400]
mask = (x - 200) ** 2 + (y - 200) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

ax2 = WordCloud(background_color="white", repeat=True, mask=mask, random_state=2,)
ax2.generate(allWordsLem)

plt.axis("off")
plt.imshow(ax2, interpolation="bilinear");

"""
******************************************************************************************************************************
Frequency Distributions
******************************************************************************************************************************
"""

words = nltk.word_tokenize(allWordsLem)
fd = FreqDist(words)
fd.most_common(3)


topWords = fd.most_common(20)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(topWords))
import seaborn as sns
sns.set_theme(style="ticks")

plt.figure(figsize=(10, 6))
ax3 = sns.barplot(y=fdist.index, x=fdist.values, palette='mako');
plt.show()

import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()

"""
******************************************************************************************************************************
Sentiment Analysis
******************************************************************************************************************************
"""
analyzer = SentimentIntensityAnalyzer()
redditComments['polarity'] = redditComments['Comment_lem'].apply(lambda x: analyzer.polarity_scores(x))

redditComments = pd.concat([redditComments.drop(['Unnamed: 0', 'Author', 'polarity', 'Comment_adj'], axis=1), redditComments['polarity'].apply(pd.Series)], axis=1)
redditComments['sentiment'] = redditComments['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')

print(redditComments.loc[redditComments['compound'].idxmax()].values)
print(redditComments.loc[redditComments['compound'].idxmin()].values)


fig, ax4 = plt.subplots(figsize=(10,6))
ax4 = sns.countplot(y="sentiment", data=redditComments, palette='mako')
ax4.set_title("Breakdown of Sentiment of Reddit Comments")
ax4.figure.tight_layout()
ax4.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\sentimentBreakdown.png', dpi= 600) 
plt.show()


"""
CRAP PLOT
ax5 = sns.lineplot(x='Date', y='compound', data=redditComments)

ax5.set(xticklabels=[]) 
ax5.set(title='Sentiment of Tweets')
ax5.set(xlabel="Time")
ax5.set(ylabel="Sentiment")
ax5.tick_params(bottom=False)
ax5.axhline(0, ls='--', c = 'grey');
"""


