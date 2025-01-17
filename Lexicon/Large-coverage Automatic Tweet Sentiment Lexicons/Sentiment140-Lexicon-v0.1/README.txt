Sentiment140 Lexicon
Version 0.1
9 April 2013
Copyright (C) 2013 National Research Council Canada (NRC)
Contact: Saif Mohammad (uvgotsaif@gmail.com)

Terms of use:
1. This lexicon can be used freely for research purposes. 
2. The papers listed below provide details of the creation and use of 
   the lexicon. If you use a lexicon, then please cite the associated 
   papers.
3. If interested in commercial use of the lexicon, send email to the 
   contact. 
4. If you use the lexicon in a product or application, then please 
   credit the authors and NRC appropriately. Also, if you send us an 
   email, we will be thrilled to know about how you have used the 
   lexicon.
5. National Research Council Canada (NRC) disclaims any responsibility 
   for the use of the lexicon and does not provide technical support. 
   However, the contact listed above will be happy to respond to 
   queries and clarifications.
6. Rather than redistributing the data, please direct interested 
   parties to this page:
   http://www.purl.com/net/lexicons 

Please feel free to send us an email:
- with feedback regarding the lexicon. 
- with information on how you have used the lexicon. 
- if interested in having us analyze your data for sentiment, emotion, 
  and other affectual information.
- if interested in a collaborative research project.

.......................................................................

SENTIMENT140 LEXICON
--------------------
The Sentiment140 Lexicon is a list of words and their associations with
positive and negative sentiment. The lexicon is distributed in three files:
unigrams-pmilexicon.txt, bigrams-pmilexicon.txt, and pairs-pmilexicon.txt.

Each line in the three files has the format:

term<tab>sentimentScore<tab>numPositive<tab>numNegative
where:
   term 
      In unigrams-pmilexicon.txt, term is a unigram (single word).
	  In bigrams-pmilexicon.txt, term is a bigram (two-word sequence).
	  A bigram has the form: "string string". The bigram was seen at least once in 
	  the source tweets from which the lexicon was created. 
	  In pairs-pmilexicon.txt, term is a unigram--unigram pair,
      unigram--bigram pair, bigram--unigram pair, or a bigram--bigram pair.
	  The pairs were generated from a large set of source tweets. Tweets were 
	  examined one at a time, and all possible unigram and bigram combinations 
	  within the tweet were chosen. Pairs with certain punctuations, @ symbols,
	  and some function words were removed.

	  
   sentimentScore is a real number. A positive score indicates positive 
      sentiment. A negative score indicates negative sentiment. The absolute 
      value is the degree of association with the sentiment.
	  The sentiment score was calculated by subtracting the pointwise mutual
	  information (PMI) score of the term with positive emoticons and the
	  PMI of the term with negative emoticons. 
	  
	  Terms with a non-zero PMI score with positive emoticons and PMI score of 0 
	  with negative emoticons were assigned a sentimentScore of 5.
	  Terms with a non-zero PMI score with negative emoticons and PMI score of 0 
	  with positive emoticons were assigned a sentimentScore of -5.

   numPositive is the number of times the term co-occurred with a positive 
      marker such as a positive emoticon or a positive emoticons.

   numNegative is the number of times the term co-occurred with a negative 
      marker such as a negative emoticon or a negative emoticons.

The Sentiment140 Lexicon was created from the Sentiment140 emoticon corpus of 1.6 million tweets.
http://help.sentiment140.com/for-students

The number of entries in:
  unigrams-pmilexicon.txt: 62,468 terms
  bigrams-pmilexicon.txt: 677,698 terms
  pairs-pmilexicon.txt: 480,010 terms

Refer to publication below for more details.

.......................................................................

PUBLICATION
-----------
Details of the lexicon can be found in the following peer-reviewed
publication:

-- In Proceedings of the seventh international workshop on Semantic 
Evaluation Exercises (SemEval-2013), June 2013, Atlanta, Georgia, USA. 

BibTeX entry:
@InProceedings{MohammadKZ2013,
  author    = {Mohammad, Saif and Kiritchenko, Svetlana and Zhu, Xiaodan},
  title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
  booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
  month     = {June},
  year      = {2013},
  address   = {Atlanta, Georgia, USA}
}
.......................................................................

