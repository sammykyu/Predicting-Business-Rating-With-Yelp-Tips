library(jsonlite)
library(NLP)
library(tm)
library(dplyr)
library(SnowballC)
library(wordcloud)
library(lattice)
library(ggplot2)
library(caret)
library(pls)
library(FactoMineR)
library(stats)
library(glmnet)

bus <- stream_in(file("yelp_academic_dataset_business.json"))
reviews <- stream_in(file("yelp_academic_dataset_review.json"))

## preprocessing
vsource <- VectorSource(reviews$text)
corpus <- Corpus(vsource)

## cleaning
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

## make a document-term matrix
dtm <- DocumentTermMatrix(corpus)
## TODO: think about using sparse matrix instead of removing sparse terms
dtmSparse <- removeSparseTerms(dtm, 0.996)
dtm2 <- as.matrix(dtmSparse)

## Find frequent terms
frequency <- colSums(dtm2)
frequency <- sort(frequency, decreasing=TRUE)
## make word cloud
words <- names(frequency)
wordcloud(words, frequency)


dtm_revs <- cbind(business_id=reviews$business_id, as.data.frame(dtm2))
## merge tips with bus
busrevs <- merge(bus[,c("business_id","stars")], dtm_revs, by="business_id")
busrevs$business_id <- NULL
