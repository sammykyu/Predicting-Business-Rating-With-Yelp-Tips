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
tips <- stream_in(file("yelp_academic_dataset_tip.json"))

## preprocessing
vsource <- VectorSource(tips$text)
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
dtmSparse <- removeSparseTerms(dtm, 0.98)
dtm2 <- as.matrix(dtmSparse)

## Find frequent terms
frequency <- colSums(dtm2)
frequency <- sort(frequency, decreasing=TRUE)
## make word cloud
words <- names(frequency)
wordcloud(words, frequency)

dtm_tips <- cbind(business_id=tips$business_id, as.data.frame(dtm2))
dtm_tips$business_id <- as.character(dtm_tips$business_id)

## merge tips with bus
#bustips <- select(bus, business_id, stars, likes, text)
bustips <- merge(bus[,c("business_id","stars")], dtm_tips, by="business_id")
bustips$business_id <- NULL
#bustips$likes <- as.numeric(bustips$likes)

## linear regression
set.seed(2046)
## partition the original training data into a training set (70%) and a validation set (30%)
inTrain <- createDataPartition(y=bustips$stars, p=0.7, list=FALSE)
training <- bustips[inTrain,]
validation <- bustips[-inTrain,]
lm.fit <- lm(stars ~., data=validation)

## ridge regression
x=model.matrix(stars~., bustips)[,-1]
y=bustips$stars
grid=10^seq(10,-2, length=100)
set.seed(1)
train=sample (1: nrow(x), nrow(x) * 0.7)
test=(-train)
y.test=y[test]

ridge.mod=glmnet(x[train ,],y[train],alpha=0, lambda =grid, thresh =1e-12)
cv.out=cv.glmnet(x[train ,],y[train],alpha=0)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
ridge.pred=predict (ridge.mod ,s=bestlam ,newx=x[test,])
mean((ridge.pred -y.test)^2)

## Lasso
lasso.mod=glmnet(x[train ,],y[ train],alpha=1, lambda =grid)
plot(lasso.mod)
set.seed(1)
cv.out=cv.glmnet(x[train ,],y[ train],alpha=1)
plot(cv.out)
bestlam =cv.out$lambda.min
lasso.pred=predict (lasso.mod ,s=bestlam ,newx=x[test ,])
mean((lasso.pred -y.test)^2)
out=glmnet (x,y,alpha=1, lambda=grid)
lasso.coef=predict (out ,type="coefficients",s= bestlam) [1:20,]
lasso.coef

## pcr
set.seed(2)
pcr.fit <- pcr(stars ~., data= bustips, scale=TRUE, validation="CV")
summary(pcr.fit)
