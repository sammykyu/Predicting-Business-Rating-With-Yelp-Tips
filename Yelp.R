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
tips <- stream_in(file("yelp_academic_dataset_tip.json"))

## preprocessing
vsource <- VectorSource(tips$text)
corpus <- Corpus(vsource)

## cleaning
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
#corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

BigramTokenizer <- function(x)
    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

dtm <- DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer))

## make a document-term matrix
# dtm <- DocumentTermMatrix(corpus)
dtmSparse <- removeSparseTerms(dtm, 0.998)
dtm2 <- as.matrix(dtmSparse)

## Find frequent terms
frequency <- colSums(dtm2)
frequency <- sort(frequency, decreasing=TRUE)
## make word cloud
words <- names(frequency)
maxlen <- ifelse(length(words) > 50, 50, length(words))
## show at most 100 most frequent used terms in a word cloud
wordcloud(words[1:maxlen], frequency[1:maxlen])

dtm_tips <- cbind(business_id=tips$business_id, as.data.frame(dtm2))

## merge tips with bus
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

## prepare data for RR and Lasso
x=model.matrix(stars~., bustips)[,-1]
y=bustips$stars
grid=10^seq(10,-2, length=100)
set.seed(1)
train=sample (1: nrow(x), nrow(x) * 0.7)
test=(-train)
y.test=y[test]

## Naive model
naive.pred <- mean(bus$stars)
mean((naive.pred -y.test)^2)


## ridge regression
RidgeRegression <- function (trainingParams) {
  ridge.mod <- glmnet(trainingParams$x.train, trainingParams$y.train, alpha=0, lambda=trainingParams$lambdas, thresh=1e-12)
  set.seed(1)
  cv.out <- cv.glmnet(trainingParams$x.train, trainingParams$y.train, alpha=0)
  bestlamda <- cv.out$lambda.min
  ridge.pred <- predict(ridge.mod, s=bestlamda, newx=trainingParams$x.test)
  mse <- mean((ridge.pred - trainingParams$y.test)^2)
  
  return(list(mod=ridge.mod, bestlamda=bestlamda, mse=mse))
}


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
lasso.pred=predict (lasso.mod ,s=bestlam ,newx=x[test,])
mean((lasso.pred -y.test)^2)
out=glmnet (x,y,alpha=1, lambda=grid)
lasso.coef2=predict (out ,type="coefficients",s= bestlam)
lasso.coef=lasso.coef2[1:dim(lasso.coef2)[1],]
lasso.coef
lasso.coef[lasso.coef>0]
length(lasso.coef[lasso.coef>0])

## pcr
set.seed(2)
pcr.fit <- pcr(stars ~., data= bustips, subset=train, scale=TRUE, validation="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP")
pcr.pred <- predict(pcr.fit, newx=x[test,],ncomp=35)
mean((pcr.pred[,1,1]-y.test)^2)

## Prcomp
pca.fit <- prcomp(x = x[train,], retx=TRUE, center = TRUE, scale=TRUE)
pca.fit
plot(pca.fit, type="l")
summary(pca.fit)
predict(pca.fit, x[test,])

## PCA
pca2 = PCA(x[train ,], graph = FALSE)
# matrix with eigenvalues
pca2$eig
# correlations between variables and PCs
pca2$var$coord
# PCs (aka scores)
head(pca2$ind$coord)
summary(pca2)
