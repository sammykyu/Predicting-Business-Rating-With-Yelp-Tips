library(jsonlite)
library(NLP)
library(tm)

bus <- stream_in(file("yelp_academic_dataset_business.json"))
tips <- stream_in(file("yelp_academic_dataset_tip.json"))

## merge tips with bus



