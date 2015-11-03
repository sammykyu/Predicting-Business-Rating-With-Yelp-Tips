memory.size(40000)

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
#library("tm.plugin.dc")
corpus <- Corpus(vsource)
#corpus <- DistributedCorpus(vsource)

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
dtmSparse <- removeSparseTerms(dtm, 0.95)
dtm2 <- as.matrix(dtmSparse)

## Find frequent terms
frequency <- colSums(dtm2)
frequency <- sort(frequency, decreasing=TRUE)
## make word cloud
words <- names(frequency)
wordcloud(words[1:100], frequency[1:100])


dtm_reviews <- cbind(business_id=reviews$business_id, as.data.frame(dtm2))
## merge tips with bus
busreviews <- merge(bus[,c("business_id","stars")], dtm_reviews, by="business_id")
busreviews$business_id <- NULL


## prepare data for RR and Lasso
x=model.matrix(stars~., busreviews)[,-1]
y=busreviews$stars
grid=10^seq(10,-2, length=100)
set.seed(1)
train=sample (1: nrow(x), nrow(x) * 0.7)
test=(-train)
y.test=y[test]

## Lasso
lasso.mod=glmnet(x[train ,],y[ train],alpha=1, lambda =grid)
plot(lasso.mod)
set.seed(1)
cv.out=cv.glmnet(x[train ,],y[ train],alpha=1)
plot(cv.out)
bestlam =cv.out$lambda.min
lasso.pred=predict (lasso.mod ,s=bestlam ,newx=x[test,])
mean((lasso.pred -y.test)^2)
out=glmnet (x,y,alpha=1, lambda=grid)
lasso.coef=predict (out ,type="coefficients",s= bestlam)[1:295,]
lasso.coef
lasso.coef[lasso.coef>0]
length(lasso.coef[lasso.coef>0])

