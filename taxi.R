################################################################################

## https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i
## Santiago Mota
## santiago_mota@yahoo.es

# Predict the destination of taxi trips based on initial partial trajectories

# The taxi industry is evolving rapidly. New competitors and technologies are 
# changing the way traditional taxi services do business. While this evolution 
# has created new efficiencies, it has also created new problems. 

# One major shift is the widespread adoption of electronic dispatch systems that 
# have replaced the VHF-radio dispatch systems of times past. These mobile data 
# terminals are installed in each vehicle and typically provide information on 
# GPS localization and taximeter state. Electronic dispatch systems make it easy 
# to see where a taxi has been, but not necessarily where it is going. In most 
# cases, taxi drivers operating with an electronic dispatch system do not 
# indicate the final destination of their current ride.

# Another recent change is the switch from broadcast-based (one to many) radio 
# messages for service dispatching to unicast-based (one to one) messages. With 
# unicast-messages, the dispatcher needs to correctly identify which taxi they 
# should dispatch to a pick up location. Since taxis using electronic dispatch 
# systems do not usually enter their drop off location, it is extremely 
# difficult for dispatchers to know which taxi to contact. 

# To improve the efficiency of electronic taxi dispatching systems it is 
# important to be able to predict the final destination of a taxi while it is in 
# service. Particularly during periods of high demand, there is often a taxi 
# whose current ride will end near or exactly at a requested pick up location 
# from a new rider. If a dispatcher knew approximately where their taxi drivers 
# would be ending their current rides, they would be able to identify which taxi 
# to assign to each pickup request.

# The spatial trajectory of an occupied taxi could provide some hints as to 
# where it is going. Similarly, given the taxi id, it might be possible to 
# predict its final destination based on the regularity of pre-hired services. 
# In a significant number of taxi rides (approximately 25%), the taxi has been 
# called through the taxi call-center, and the passenger’s telephone id can be 
# used to narrow the destination prediction based on historical ride data 
# connected to their telephone id.

# In this challenge, we ask you to build a predictive framework that is able to 
# infer the final destination of taxi rides in Porto, Portugal based on their 
# (initial) partial trajectories. The output of such a framework must be the 
# final trip's destination (WGS84 coordinates).

# This is the first of two data science challenges that share the same dataset. 
# The Taxi Service Trip Time competition predicts the total time of taxi rides.
# https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii

# This competition is affiliated with the organization of ECML/PKDD 2015.

# Started: 7:04 pm, Monday 20 April 2015 UTC
# Ends: 11:59 pm, Wednesday 1 July 2015 UTC (72 total days)
# Points: this competition awards 0.5X ranking points
# Tiers: this competition counts towards tiers 

## Data information

# File Name       Available Formats
# evaluation_script 	.r (1.38 kb)
# metaData_taxistandsID_name_GPSlocation.csv 	.zip (2.41 kb)
# sampleSubmission.csv 	.zip (994 b)
# test.csv 	.zip (85.59 kb)
# train.csv 	.zip (508.89 mb)


# I. Training Dataset

# We have provided an accurate dataset describing a complete year (from 
# 01/07/2013 to 30/06/2014) of the trajectories for all the 442 taxis running 
# in the city of Porto, in Portugal (i.e. one CSV file named "train.csv"). These 
# taxis operate through a taxi dispatch central, using mobile data terminals 
# installed in the vehicles. We categorize each ride into three categories: 
# A) taxi central based, B) stand-based or C) non-taxi central based. For the 
# first, we provide an anonymized id, when such information is available from 
# the telephone call. The last two categories refer to services that were 
# demanded directly to the taxi drivers on a B) taxi stand or on a C) random 
# street.

# Each data sample corresponds to one completed trip. It contains a total of
# 9 (nine) features, described as follows:

# 1. TRIP_ID: (String) It contains an unique identifier for each trip;
# 2. CALL_TYPE: (char) It identifies the way used to demand this service. It may 
#    contain one of three possible values:
#    - ‘A’ if this trip was dispatched from the central;
#    - ‘B’ if this trip was demanded directly to a taxi driver on a specific stand;
#    - ‘C’ otherwise (i.e. a trip demanded on a random street).
# 3. ORIGIN_CALL: (integer) It contains an unique identifier for each phone 
#    number which was used to demand, at least, one service. It identifies the 
#    trip’s customer if CALL_TYPE=’A’. Otherwise, it assumes a NULL value;
# 4. ORIGIN_STAND: (integer): It contains an unique identifier for the taxi 
#    stand. It identifies the starting point of the trip if CALL_TYPE=’B’. 
#    Otherwise, it assumes a NULL value;
# 5. TAXI_ID: (integer): It contains an unique identifier for the taxi driver 
#    that performed each trip;
# 6. TIMESTAMP: (integer) Unix Timestamp (in seconds). It identifies the trip’s 
#    start; 
# 7. DAYTYPE: (char) It identifies the daytype of the trip’s start. It assumes 
#    one of three possible values:
#    - ‘B’ if this trip started on a holiday or any other special day (i.e. 
#          extending holidays, floating holidays, etc.);
#    - ‘C’ if the trip started on a day before a type-B day;
#    - ‘A’ otherwise (i.e. a normal day, workday or weekend).
# 8. MISSING_DATA: (Boolean) It is FALSE when the GPS data stream is complete 
#    and TRUE whenever one (or more) locations are missing
# 9. POLYLINE: (String): It contains a list of GPS coordinates (i.e. WGS84 
#    format) mapped as a string. The beginning and the end of the string are 
#    identified with brackets (i.e. [ and ], respectively). Each pair of 
#    coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]. 
#    This list contains one pair of coordinates for each 15 seconds of trip. The 
#    last list item corresponds to the trip’s destination while the first one 
#    represents its start;

# New variables
#    Date: class Date
#    Time: Class POSIXct
#    Duration: seconds


# II. Testing

# Five test sets will be available to evaluate your predictive framework (in one 
# single CSV file named "test.csv"). Each one of these datasets refer to trips 
# that occurred between 01/07/2014 and 31/12/2014. Each one of these data sets 
# will provide a snapshot of the current network status on a given timestamp. It 
# will provide partial trajectories for each one of the on-going trips during 
# that specific moment.

# The five snapshots included on the test set refer to the following timestamps:

# 14/08/2014 18:00:00
# 30/09/2014 08:30:00
# 06/10/2014 17:45:00
# 01/11/2014 04:00:00
# 21/12/2014 14:30:00

# III. Sample Submission Files

# File sampleSubmission.csv uses the location of Porto main Avenue, in downtown 
# (i.e. Avenida dos Aliados). 

# IV. Other Files

# Along with these two files, we have also provided two additional files. One 
# contains meta data regarding the taxi stands metaData_taxistandsID_name_GPSlocation.csv 
# including id and location.

# The second one includes an evaluation script for both problems developed in 
# the R language ("evaluation_script.r").


# Evaluation

# The evaluation metric for this competition is the Mean Haversine Distance. The 
# Haversine Distance is commonly used in navigation. It measures distances 
# between two points on a sphere based on their latitude and lagitude.

# Let P1,P2 be two points which location is given by the WGS84 coordinates 
# (l1,L1) and (l2,L2), respectively. The Harvesine Distance between the two 
# locations P1, P2 (HDist) can be computed as follows

# Harvesine Distance Equation

# where R is the sphere's radius. In our case, it should be replaced by the 
# Earth's radius in the desired metric (e.g., kilometers).

# Submission Format

# For every trip in the dataset, submission files should contain three columns: 
# TRIP_ID, LATITUDE, and LONGITUDE. TRIP_ID represents the ID of the trip for 
# which you are predicting the destination (i.e. a string). The 
# LATITUDE/LONGITUDE represent the location's coordinates (using WGS84 format) 
# of your predicted destination

# The file should contain a header and have the following format:

# TRIP_ID, LATITUDE, LONGITUDE
# T1, 41.146504,-8.611317
# T2, 42.230000,-8.629454
# T10, 42.110000,-8.721111

################################################################################
## Some initial work
Sys.setenv(LANGUAGE="en")
set.seed(1967)

# Info session
sessionInfo()

# Show elements working directory
ls()

# Gets you the current working directory
getwd()                    

# Lists all the files present in the current working directory
dir()

# Updates all packages
update.packages() 

################################################################################

library(rjson)
library(data.table)

# Control the number of trips read for training (all=-1)
# Control the number of closest trips used to calculate trip duration
# N_read <- 100000
# N_trips <- 1000
N_read <- -1
# N_trips <- 10000

### Get starting & ending longitude and latitude
get_coordinate <- function(row){
      lonlat    <- fromJSON(row)
      snapshots <- length(lonlat)  
      start     <- lonlat[[1]]
      end       <- lonlat[[snapshots]]
      return(list(start[1], start[2], end[1], end[2], snapshots))
} 

HaversineDistance <- function(lat1, lon1, lat2, lon2)
{
      # returns the distance in m
      REarth <- 6371000
      lat  <- abs(lat1-lat2)*pi/180
      lon  <- abs(lon1-lon2)*pi/180
      lat1 <- lat1*pi/180
      lat2 <- lat2*pi/180
      a    <- sin(lat/2)*sin(lat/2)+cos(lat1)*cos(lat2)*sin(lon/2)*sin(lon/2)
      d    <- 2*atan2(sqrt(a), sqrt(1-a))
      d    <- REarth*d
      return(d)
}

RMSE <- function(pre, real)
{
      return(sqrt(mean((pre-real)*(pre-real))))
}

meanHaversineDistance <- function(lat1, lon1, lat2, lon2)
{
      return(mean(HaversineDistance(lat1, lon1, lat2, lon2)))
}

# Get Haversine distance
get_dist <- function(lon1, lat1, lon2, lat2) {  
      lon_diff <- abs(lon1-lon2)*pi/360
      lat_diff <- abs(lat1-lat2)*pi/360
      a        <- sin(lat_diff)^2 + cos(lat1) * cos(lat2) * sin(lon_diff)^2  
      d        <- 2*6371000*atan2(sqrt(a), sqrt(1-a))
      return(d)
}

################################################################################
# loading all zip files from kaggle
# Load data

library(readr)
library(rjson)

# There is an error on ID 41. Latitude and Longitude without space. 
# Changed with Libreoffice
meta_data <- read.csv('./data/metaData_taxistandsID_name_GPSlocation.csv', 
                      stringsAsFactors=FALSE, dec=".")
# test      <- read.csv('./data/test.csv')
# train     <- read.csv('./data/train.csv')

# Use the readr package in order to use zip data and save disk space
test  <- read_csv("./data/test.csv.zip")
train <- read_csv("./data/train.csv.zip")

summary(meta_data)
summary(test)
summary(train)

# Plot latitude, longitude
plot(meta_data[, c(4, 3)])

# Change variable names
names(test)      <- tolower(names(test))
names(train)     <- tolower(names(train))
names(meta_data) <- tolower(names(meta_data))

# Change meta_dat variable id name to origin_stand
names(meta_data)[1] <- 'origin_stand'

# Change missing_data zand call_type from character to factor
train$missing_data <- as.factor(train$missing_data)
test$missing_data  <- as.factor(test$missing_data)
train$call_type    <- as.factor(train$call_type)
test$call_type     <- as.factor(test$call_type)

# Look for missing data
table(train$missing_data)
table(test$missing_data)

# Some analysis
table(test$taxi_id, useNA="ifany")[order(table(test$taxi_id, useNA="ifany"))]
table(train$taxi_id)[order(table(train$taxi_id))]
table(train$call_type, useNA="ifany")
# A      B      C 
# 364770 817881 528019
table(test$call_type, useNA="ifany")
#  A      B      C 
# 72    123    125
table(train$origin_call, useNA="ifany")[order(table(train$origin_call, 
                                                    useNA="ifany"))]

table(train$origin_stand, useNA="ifany")[order(table(train$origin_stand, 
                                                     useNA="ifany"))]
table(test$origin_stand, useNA="ifany")[order(table(test$origin_stand, 
                                                    useNA="ifany"))]

table(test$taxi_id, useNA="ifany")[order(table(test$taxi_id, useNA="ifany"))]
table(train$taxi_id, useNA="ifany")[order(table(train$taxi_id, useNA="ifany"))]

# There is only data on day type A
table(test$day_type, useNA="ifany")[order(table(test$day_type, useNA="ifany"))]
table(train$day_type, useNA="ifany")[order(table(train$day_type, useNA="ifany"))]

# Delete day_type
test$day_type  <- NULL
train$day_type <- NULL

# There is only 10 over 1710670 missing_data on train and 0 over 320 on test

max(train$date, na.rm=TRUE)
min(train$date, na.rm=TRUE)
max(train$date, na.rm=TRUE)-min(train$date, na.rm=TRUE)
hist(train$date, breaks=343)
table(train$date, useNA="ifany")
table(test$date, useNA="ifany")
plot(test$timestamp)
# Test data contains only 5 days

# Make a new variable with time in POSIXct format
library(RgoogleMaps)
library(colorRamps)
library(tm)
library(chron)
format_alg <- function(i, n)
{
      s <- sprintf("%d", n)
      while(nchar(s)<i)
      {
            s <- sprintf("0%s", s)
      }
      return(s)
}

# Split the 'date' variable in other two: 'year' and 'month'
# dates <- strsplit(as.character(contributorsdf$date), "-")
# contributorsdf$year  <- sapply(dates, function(x) x[1])
# contributorsdf$month <- sapply(dates, function(x) x[2])

miliseconds_to_date <- function(dt)
{
      # Obtain the mdy$year, mdy$month and mdy$day from timestamp
      mdy <- month.day.year((dt/3600/24))
      
      final <- as.Date(x=paste(mdy$year, "/", mdy$month, "/", mdy$day, sep=""), 
                       format="%Y/%m/%d")
      
      return(final)
}

test$date  <- miliseconds_to_date(test$timestamp)
train$date <- miliseconds_to_date(train$timestamp)

test$time  <- as.POSIXct(test$timestamp, origin="1970-01-01")
train$time <- as.POSIXct(train$timestamp, origin="1970-01-01")

test$weekday <- factor(weekdays(test$date), levels=c("lunes", "martes", 
                                                     "miércoles", "jueves",
                                                     "viernes", "sábado",
                                                     "domingo"))

train$weekday <- factor(weekdays(train$date), levels=c("lunes", "martes", 
                                                       "miércoles", "jueves",
                                                       "viernes", "sábado",
                                                       "domingo"))

library(lubridate)
test$hour  <- hour(test$time)
train$hour <- hour(train$time)

save(test, file="./data/test.RData")
save(train, file="./data/train.RData")

###############################################################################
# Read
library(data.table)
train_DT <- fread('./data/train.csv', select=c('TRIP_ID', 'POLYLINE'), 
                  stringsAsFactors=F, nrows=N_read)
test_DT  <- fread('./data/test.csv', select=c('TRIP_ID', 'POLYLINE'), 
                  stringsAsFactors=F)

summary(test_DT)
summary(train_DT)

train_DT[, index:=-seq(.N, 1, -1)]
train_index <- train_DT$index
train_DT <- train_DT[POLYLINE!='[]']
test_DT[, index:=1:.N]
setkey(train_DT, index)
setkey(test_DT, index)

# Watch out. trip_id value is not unique, so we include index

# Get starting & ending position from POLYLINE
train_DT[, c('initial_longitude', 'initial_latitude', 'final_longitude', 
             'final_latitude', 'snapshots'):=get_coordinate(POLYLINE), by=index]
test_DT[, c('initial_longitude', 'initial_latitude', 'final_longitude', 
            'final_latitude', 'snapshots'):=get_coordinate(POLYLINE), by=index]

# Delete POLYLINE values
train_DT[, POLYLINE:=NULL]
test_DT[, POLYLINE:=NULL]

# Make a new dataset througth data table
train2 <- as.data.frame(train_DT)
test2  <- as.data.frame(test_DT)

# Change variable names
names(test2)  <- tolower(names(test2))
names(train2) <- tolower(names(train2))

save(test2, file="./data/test2.RData")
save(train2, file="./data/train2.RData")

################################################################################
load("./data/test.RData")
load("./data/train.RData")

# load("./data/train_trip.RData")
load("./data/test2.RData")
load("./data/train2.RData")

test$index  <- test2$index
train$index <- train_index
test3  <- merge(test[, c(1:7, 9:13)], test2, by.x="index", by.y="index")
train2$trip_id <- as.numeric(train2$trip_id)
index <- (1:nrow(train))[train$polyline!="[]"]
train3 <- merge(train[, c(1:7, 9:13)], train2, by.x="index", 
                by.y="index")

test3$trip_id.y  <- NULL
train3$trip_id.y <- NULL

summary(test3)
summary(train3)

test3$distance <- HaversineDistance(test3$initial_latitude, 
                                    test3$initial_longitude,
                                    test3$final_latitude,
                                    test3$final_longitude)

train3$distance <- HaversineDistance(train3$initial_latitude, 
                                     train3$initial_longitude,
                                     train3$final_latitude,
                                     train3$final_longitude)

# No movement. Test=2. Train=30609
test3[test3$snapshots==1, ]
train3[train3$snapshots==1, ]

# No distance. Test=2. Train=20974
test3[test3$distance==0, ]
train3[train3$distance==0, ]

# Distance <100m. Test=24. Train=63405 
test3$distance[test3$distance<100]
train3$distance[train3$distance<100]
sum(test3$distance<100)
# [1] 24
sum(train3$distance<100)
# [1] 63405

# Long distances
test3$distance[test3$distance>100000]
train3$distance[train3$distance>200000]

# 5 day test dates. 365 days train dates.
date_test  <- unique(test3$date)
date_train <- unique(train3$date)

# Weekdays on train and test
table(test3$weekday)
table(train3$weekday)

table(test$weekday)/nrow(test)
table(train$weekday)/nrow(train)

# On test dataset there is no values on miercoles or viernes
weekdays(date_test)
# [1] "jueves"  "martes"  "lunes"   "sábado"  "domingo"

# Hour analysis
table(test$hour)[order(table(test$hour))]
table(train$hour)[order(table(train$hour))]

table(test$hour)/nrow(test)
table(train$hour)/nrow(train)

# Time structure of test data. 5 days. 
# Jueves 19:00, martes 10:00, lunes 19:00, sabado 04:00 & domingo 15:00
plot(test$hour)
plot(test$date)
plot(test$weekday)

# More information
plot(test3$initial_longitude, test3$initial_latitude)
plot(test3$final_longitude, test3$final_latitude)
hist(train3$snapshots)
train3$duration[train3$snapshots>20000]
table(test3$date, useNA="ifany")
plot(table(train3$snapshots))
plot(table(test3$snapshots))
plot(test3$distance/test3$snapshots)
plot(train3$distance/train3$snapshots)

save(test3, file="./data/test3.RData")
save(train3, file="./data/train3.RData")

################################################################################
# Making the final datasets

load("./data/test3.RData")
plot(test3$timestamp)
plot(test3$weekday)
load("./data/train3.RData")
test4  <- test3
train4 <- train3
plot(train4$timestamp)
plot(train4$hour)
hist(train4$hour)
hist(test4$hour)
plot(test4$timestamp)
hist(test4$timestamp)
plot(test4$time)

max(test4$time)
min(test4$time)
max(train4$time)
min(train4$time)

# We aggregate minute values in 15min groups
test4$minute  <- hour(test4$time)*60+(minute(test4$time)%/%15)*15
train4$minute <- hour(train4$time)*60+(minute(train4$time)%/%15)*15

# Some minute variable analysis
plot(test4$minute)
hist(test4$minute)

plot(train4$minute)
hist(train4$minute)
table(train4$minute)
plot(table(train4$minute))
plot(table(train4$minute), col=train4$weekday)
table(train4$minute, train4$weekday)

# Generate data
library(ggplot2)
g <- ggplot(train4, aes(minute))
g + geom_bar()

ggplot(train4, aes(minute, fill=weekday)) + geom_bar()

ggplot(train4, aes(minute)) + geom_bar() + facet_wrap(~ weekday)

## Unusual cases

# Trips of legth less or iqual to 1 minute. Train: 46202. Test: 21
sum(train4$snapshots<5)
sum(test4$snapshots<5)
train4[train4$snapshots<5, ]

# Distance less than 100 meters. Train: 63405. Test: 24
sum(train4$distance<100)
sum(test4$distance<100)

# Speed greater than 30 m/s. Train: 1781. Test: 1 
train_speed <- train4$distance/((train4$snapshots-1)*15)
sum(train_speed>30, na.rm=TRUE)
test_speed <- test4$distance/((test4$snapshots-1)*15)
sum(test_speed>30, na.rm=TRUE)

save(test4, file="./data/test4.RData")
save(train4, file="./data/train4.RData")

###############################################################################
# More information on test4 and train4 datasets

load(file="./data/test4.RData")
load(file="./data/train4.RData")

# Initial and final trip positions of test dataset
qplot(initial_longitude, initial_latitude, data=test4)
qplot(final_longitude, final_latitude, data=test4)

hist(test4$initial_longitude)
hist(test4$initial_latitude)

hist(train4$initial_longitude)
hist(train4$initial_latitude)

summary(test4$initial_longitude)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -8.689  -8.628  -8.612  -8.616  -8.603  -8.547

summary(test4$initial_latitude)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 41.09   41.15   41.15   41.16   41.17   41.24

summary(train4$initial_longitude)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -36.910  -8.629  -8.613  -8.617  -8.604  52.900 
summary(train4$initial_latitude)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 31.99   41.15   41.15   41.16   41.16   51.04

# Area: [-8.629, 41.15]-[-8.603, 41.17] 

HaversineDistance(-8.629, 41.15, -8.603, 41.15)
# [1] 2891.068
HaversineDistance(-8.629, 41.15, -8.629, 41.17)
# [1] 2198.725

################################################################################
## Actual models 

##### h2o ver 3.0 deep learning #####
load(file="./data/train4.RData")
load(file="./data/test4.RData")
library(h2o)
localH2O <- h2o.init(nthread=8, max_mem_size="16g")

train_hex <- as.h2o(localH2O, train4[, c("weekday", "minute", "initial_longitude", 
                                         "initial_latitude", "call_type",
                                         "taxi_id", "snapshots")])
test_hex  <- as.h2o(localH2O, test4[, c("weekday", "minute", "initial_longitude", 
                                        "initial_latitude", "call_type",
                                        "taxi_id")])
my_X <- 1:5
my_Y <- "snapshots"
mod_fit039 <- h2o.deeplearning(x=my_X, y=my_Y, training_frame=train_hex, 
                               activation="Tanh", hidden=c(512, 256, 128),
                               epochs=32, variable_importances=T)

dl_VI <- mod_fit039@model$varimp
print(dl_VI)
plot(dl_VI)

# Prediction
pred039 <- as.data.frame(h2o.predict(mod_fit039, test_hex))$predict
pred039
save(mod_fit039, file="./pred/mod_fit039.RData")


##### h2o ver 3.0 randomForest #####
load(file="./data/train4.RData")
load(file="./data/test4.RData")
library(h2o)
localH2O <- h2o.init(nthread=8, max_mem_size="18g")

train_hex <- as.h2o(localH2O, train4[, c("weekday", "minute", "initial_longitude", 
                                         "initial_latitude", "call_type",
                                         "taxi_id", "snapshots")])
test_hex  <- as.h2o(localH2O, test4[, c("weekday", "minute", "initial_longitude", 
                                        "initial_latitude", "call_type",
                                        "taxi_id")])
my_X <- 1:6
my_Y <- "snapshots"
mod_fit050 <- h2o.randomForest(x=my_X, y=my_Y, training_frme=train_hex, 
                               ntrees = 3100, max_depth = 26, min_rows = 1,
                               nbins = 20, seed = 1967)

dl_VI <- mod_fit050@model$varimp
print(dl_VI)
plot(dl_VI)

# Prediction
pred050 <- as.data.frame(h2o.predict(mod_fit050, test_hex))$predict
pred050
save(mod_fit050, file="./pred/mod_fit050.RData")


##### Fit a gbm h2o model (3.0) #####
load(file="./data/train4.RData")
load(file="./data/test4.RData")

library(h2o)
localH2O <- h2o.init(nthread=8, max_mem_size="21g")

train_hex <- as.h2o(localH2O, train4[, c("weekday", "minute", "initial_longitude", 
                                         "initial_latitude", "call_type",
                                         "taxi_id", "snapshots")])
test_hex  <- as.h2o(localH2O, test4[, c("weekday", "minute", "initial_longitude", 
                                        "initial_latitude", "call_type",
                                        "taxi_id")])
my_X <- 1:6
my_Y <- "snapshots"
# Changed to use h2o 3.0
mod_fit051 <- h2o.gbm(
      x = my_X,
      y = my_Y,
      training_frame=train_hex,
      ntrees = 10000,
      max_depth = 15,
      min_rows = 10,
      learn_rate = 0.005,
      nbins = 20,
      validation_frame = NULL,
      balance_classes = FALSE,
      max_after_balance_size = 1,
      seed = 1967)

dl_VI <- mod_fit051@model$varimp
print(dl_VI)
plot(dl_VI)

# Prediction
pred051 <- as.data.frame(h2o.predict(mod_fit051, test_hex))$predict
pred051
plot(pred051)
save(mod_fit051, file="./pred/mod_fit051.RData")


##### Fit a gbm h2o model (3.0) #####
load(file="./data/train4.RData")
load(file="./data/test4.RData")

library(h2o)
localH2O <- h2o.init(nthread=4, max_mem_size="12g")

train_hex <- as.h2o(localH2O, train4[, c("weekday", "minute", "initial_longitude", 
                                         "initial_latitude", "call_type",
                                         "taxi_id", "snapshots")])
test_hex  <- as.h2o(localH2O, test4[, c("weekday", "minute", "initial_longitude", 
                                        "initial_latitude", "call_type",
                                        "taxi_id")])
my_X <- 1:6
my_Y <- "snapshots"
# Changed to use h2o 3.0
mod_fit057 <- h2o.gbm(
      x = my_X,
      y = my_Y,
      training_frame=train_hex,
      ntrees = 10000,
      max_depth = 20,
      min_rows = 10,
      learn_rate = 0.005,
      nbins = 25,
      validation_frame = NULL,
      balance_classes = FALSE,
      max_after_balance_size = 1,
      seed = 1967)

dl_VI <- mod_fit057@model$varimp
print(dl_VI)
plot(dl_VI)

# Prediction
pred057 <- as.data.frame(h2o.predict(mod_fit057, test_hex))$predict
pred057
plot(pred057)
save(mod_fit057, file="./pred/mod_fit057.RData")



################################################################################
## Submissions

# Other h2o model
pred039 <- as.data.frame(h2o.predict(mod_fit039, test_hex))$predict
pred039_test_time <- data.frame(TRIP_ID=test4$trip_id, 
                                TRAVEL_TIME=(round(pred039)-1)*15)

pred039_test_time$TRAVEL_TIME - (test4$snapshots-1)*15
negative <- (pred039_test_time$TRAVEL_TIME - (test4$snapshots-1)*15)<0
sum(negative)
pred039_test_time$TRAVEL_TIME[negative] <- ((test4$snapshots[negative]-1)*15)*1.45

write.csv(pred039_test_time, './pred/pred039_test_time.csv', row.names=F)
# Result: 0.57524

# Other h2o random forest
pred050 <- as.data.frame(h2o.predict(mod_fit050, test_hex))$predict
pred050_test_time <- data.frame(TRIP_ID=test4$trip_id, 
                                TRAVEL_TIME=(round(pred050)-1)*15)

pred050_test_time$TRAVEL_TIME - (test4$snapshots-1)*15
negative <- (pred050_test_time$TRAVEL_TIME - (test4$snapshots-1)*15)<0
sum(negative)
pred050_test_time$TRAVEL_TIME[negative] <- ((test4$snapshots[negative]-1)*15)*1.45

write.csv(pred050_test_time, './pred/pred050_test_time.csv', row.names=F)
# Result: 0.56240

# Other h2o random forest
pred051 <- as.data.frame(h2o.predict(mod_fit051, test_hex))$predict
pred051_test_time <- data.frame(TRIP_ID=test4$trip_id, 
                                TRAVEL_TIME=(round(pred051)-1)*15)

pred051_test_time$TRAVEL_TIME - (test4$snapshots-1)*15
negative <- (pred051_test_time$TRAVEL_TIME - (test4$snapshots-1)*15)<0
sum(negative)
pred051_test_time$TRAVEL_TIME[negative] <- ((test4$snapshots[negative]-1)*15)*1.45

write.csv(pred051_test_time, './pred/pred051_test_time.csv', row.names=F)
# Result: 0.54445

# Other h2o random forest
pred057 <- as.data.frame(h2o.predict(mod_fit057, test_hex))$predict
pred057_test_time <- data.frame(TRIP_ID=test4$trip_id, 
                                TRAVEL_TIME=(round(pred057)-1)*15)

pred057_test_time$TRAVEL_TIME - (test4$snapshots-1)*15
negative <- (pred057_test_time$TRAVEL_TIME - (test4$snapshots-1)*15)<0
sum(negative)
pred057_test_time$TRAVEL_TIME[negative] <- ((test4$snapshots[negative]-1)*15)*1.45

write.csv(pred057_test_time, './pred/pred057_test_time.csv', row.names=F)
# Result: 0.54846

# Combine models
pred051_test_time <- read.csv(file='./pred/pred051_test_time.csv')
pred050_test_time <- read.csv(file='./pred/pred050_test_time.csv')
pred057_test_time <- read.csv(file='./pred/pred057_test_time.csv')
pred039_test_time <- read.csv(file='./pred/pred039_test_time.csv')

submissions <- data.frame(model = c(51, 50, 57, 39), 
                          public_score = c(0.54445, 0.56240, 0.54846, 0.57524),
                          type = c("h2o gbm", "h20 RF", "h2o gbm", "h2o DL"), 
                          weight = c(8.5, .5, .5, .5))

submissions$weight = submissions$weight / sum(submissions$weight)

pred058_test_time <- pred050_test_time

combine <- pred051_test_time$TRAVEL_TIME*submissions$weight[1] + 
      pred050_test_time$TRAVEL_TIME*submissions$weight[2] +
      pred057_test_time$TRAVEL_TIME*submissions$weight[3] + 
      pred039_test_time$TRAVEL_TIME*submissions$weight[4] 

combine
plot(combine - pred051_test_time$TRAVEL_TIME)

pred058_test_time$TRAVEL_TIME <- combine

write.csv(pred058_test_time, './pred/pred058_test_time.csv', row.names=F)
# 0.54401


################################################################################
## Scripts from other authors
################################################################################
## script_aux.r at Kaggle contest

#USAGE
#
#FUNCTION PARAMETERS: @filename
#@submission: path+filename containing trajectories in CSV format
process.data <- function(filename)
{
      libraries()
      dt <- read.csv2(filename, sep=",")
      
      dt$CALL_TYPE    <- as.character(dt$CALL_TYPE)
      dt$DAY_TYPE     <- as.character(dt$DAY_TYPE)
      dt$MISSING_DATA <- as.character(dt$MISSING_DATA)
      dt$POLYLINE     <- as.character(dt$POLYLINE)
      str(dt)
      print(unique(dt$MISSING_DATA))
      
      my_poly <- dt$POLYLINE[2]
      print(my_poly)
      print("")
      i <- 1
      
      for (mp in c(1:length(dt$POLYLINE)))
      {
            print(mp)
            print(miliseconds_to_date(dt$TIMESTAMP[mp]))
            my_poly <- dt$POLYLINE[mp]
            if(nchar(my_poly)>2)
            {
                  ab <- unlist(strsplit(my_poly,"[]]"))
                  polyline <- c(substr(ab[1],3,nchar(ab[1])),sapply(ab[2:(length(ab)-1)],my.substr))
                  
                  lat <- c(as.numeric(sapply(polyline,getlat)))
                  lon <- c(as.numeric(sapply(polyline,getlon)))
                  
                  create_map(length(dt$POLYLINE),lat,lon)
            }
            print(mp)
            print(miliseconds_to_date(dt$TIMESTAMP[mp]))
            x <- scan()
      }
}

miliseconds_to_date <- function(dt)
{
      mdy <- month.day.year((dt/3600/24))
      
      dt <- dt%%(3600*24)
      h  <- floor(dt/3600)
      dt <- dt-(h*3600)
      m  <- floor(dt/60)
      dt <- dt-(m*60)
      s  <- round(dt)
      
      final <- sprintf("%s/%s/%s %s:%s:%s", format_alg(4, mdy$year), 
                       format_alg(2, mdy$month), format_alg(2, mdy$day),
                       format_alg(2, h), format_alg(2, m), format_alg(2, s))
      return(final)
}

create_map <- function(ntraj, lat, lon, zoom=13)
{
      
      #random colors
      colfunc <- colorRampPalette(c("green","red"))
      
      center <- c(mean(lat), mean(lon))
      print(sprintf("Zoom: %d",zoom))
      MyMap <- GetMap(center="Porto", zoom=zoom, GRAYSCALE=FALSE, 
                      destfile = "MyTile3.png");
      
      point_size<-1.0
      
      print(lat)
      print(lon)
      tmp <- PlotOnStaticMap(MyMap, lat=lat, lon=lon, cex=point_size,pch=20, 
                             col=colfunc(length(lat)), add=FALSE)
      str(tmp)
}

load.lib <- function(libT, l=NULL)
{
      lib.loc <- l
      print(lib.loc)
      
      if (length(which(installed.packages(lib.loc=lib.loc)[,1]==libT))==0)
      {
            install.packages(libT, lib=lib.loc,repos='http://cran.us.r-project.org')
      }
}

format_alg <- function(i,n)
{
      s <- sprintf("%d", n)
      while(nchar(s)<i)
      {
            s <- sprintf("0%s", s)
      }
      return(s)
}

libraries <- function()
{
      load.lib("RgoogleMaps")
      library(RgoogleMaps)
      load.lib("colorRamps")
      library(colorRamps)
      load.lib("tm")
      library(tm)
      load.lib("chron")
      library(chron)
}

my.substr <- function(ele)
{
      ab<-substr(ele,4,nchar(ele))
      if (substr(ab,1,1)==' ')
            return(substr(ab,3,nchar(ab)))
      return(ab)
}

getlat <- function(ele)
{
      return(as.numeric(unlist(strsplit(ele,"[,]"))[2]))
}

getlon <- function(ele)
{
      return(as.numeric(unlist(strsplit(ele,"[,]"))[1]))
}

################################################################################
## evaluation_script.r at Kaggle contest
HaversineDistance <- function(lat1, lon1, lat2, lon2)
{
      #returns the distance in km
      REarth <- 6371
      lat  <- abs(lat1-lat2)*pi/180
      lon  <- abs(lon1-lon2)*pi/180
      lat1 <- lat1*pi/180
      lat2 <- lat2*pi/180
      a    <- sin(lat/2)*sin(lat/2)+cos(lat1)*cos(lat2)*sin(lon/2)*sin(lon/2)
      d    <- 2*atan2(sqrt(a), sqrt(1-a))
      d    <- REarth*d
      return(d)
}

RMSE <- function(pre, real)
{
      return(sqrt(mean((pre-real)*(pre-real))))
}

meanHaversineDistance <- function(lat1, lon1, lat2, lon2)
{
      return(mean(HaversineDistance(lat1, lon1, lat2, lon2)))
}

# USAGE
#
# FUNCTION PARAMETERS: @submission, @answers
# @submission: path+filename containing the answers to submit in CSV format
# @answers: path+filename containing the answers to evaluate the submission in CSV format

travelTime.PredictionEvaluation <- function(submission, answers)
{
      dt      <- read.csv(submission)
      tt_sub  <- dt[, 2]
      dt      <- read.csv(answers)
      tt_real <- dt[, 2]
      return (RMSE(tt_sub, tt_real))
}

# USAGE
#
# FUNCTION PARAMETERS: @submission, @answers
# @submission: path+filename containing the answers to submit in CSV format
# @answers: path+filename containing the answers to evaluate the submission in CSV format

destinationMining.Evaluation <- function(submission, answers)
{
      dt       <- read.csv(submission)
      lat_sub  <- dt[, 2]
      lon_sub  <- dt[, 3]
      dt       <- read.csv(answers)
      lat_real <- dt[, 2]
      lon_real <- dt[, 3]
      return (meanHaversineDistance(lat_sub, lon_sub, lat_real, lon_real))
}

################################################################################
## https://www.kaggle.com/benhamner/pkdd-15-predict-taxi-service-trajectory-i/last-location-benchmark
## Last Location Benchmark

library(readr)
library(rjson)

train <- read_csv("./data/train.csv.zip")
test  <- read_csv("./data/test.csv.zip")

positions <- function(row) as.data.frame(do.call(rbind, fromJSON(row$POLYLINE)))
last_position <- function(row) tail(positions(row), 1)

submission <- test["TRIP_ID"]

for (i in 1:nrow(test)) {
      print(i)
      pos <- last_position(test[i, ])
      submission[i, "LATITUDE"]  <- pos[2]
      submission[i, "LONGITUDE"] <- pos[1]
}

write_csv(submission, "last_location_benchmark.csv")

################################################################################
## Benchmark time elapsed
## https://www.kaggle.com/benhamner/pkdd-15-taxi-trip-time-prediction-ii/max-time-elapsed-mean-time-benchmark

# This benchmark predicts the maximum of the
# time that's elapsed so far in the trip and
# the mean time in the training set as the
# test trip duration

library(readr)
library(rjson)

test  <- read_csv("../input/test.csv.zip")
mean_train_time <- 660

positions <- function(row) as.data.frame(do.call(rbind, fromJSON(row$polyline)))

submission <- test["trip_id"]
names(submission)[1] <- "TRIP_ID"

for (i in 1:nrow(test)) {
      submission$TRAVEL_TIME[i] <- max(15*(nrow(positions(test[i, ]))-1), mean_train_time)
}

write_csv(submission, "max_time_elapsed_mean_time_benchmark.csv")

################################################################################
## https://www.kaggle.com/blobs/download/forum-message-attachment-files/2526/BeatTheBenchmark_loc.R
library(rjson)
library(data.table)

### Control the number of trips read for training (all=-1)
### Control the number of closest trips used to calculate trip duration
# N_read <- 100000
# N_trips <- 1000
N_read <- -1
N_trips <- 10000

### Get starting & ending longitude and latitude
get_coordinate <- function(row){
      lonlat    <- fromJSON(row)
      snapshots <- length(lonlat)  
      start     <- lonlat[[1]]
      end       <- lonlat[[snapshots]]
      return(list(start[1], start[2], end[1], end[2], snapshots))
} 

### Get Haversine distance
get_dist <- function(lon1, lat1, lon2, lat2) {  
      lon_diff <- abs(lon1-lon2)*pi/360
      lat_diff <- abs(lat1-lat2)*pi/360
      a        <- sin(lat_diff)^2 + cos(lat1) * cos(lat2) * sin(lon_diff)^2  
      d        <- 2*6371*atan2(sqrt(a), sqrt(1-a))
      return(d)
}

### Read
train <- fread('./data/train.csv', select=c('POLYLINE'), stringsAsFactors=F, nrows=N_read)
test <- fread('./data/test.csv', select=c('TRIP_ID', 'POLYLINE'), stringsAsFactors=F)
train <- train[POLYLINE!='[]']
train[, r:=-seq(.N, 1, -1)]
test[, r:=1:.N]
setkey(train, r)
setkey(test, r)

### Get starting & ending position from POLYLINE
train[, c('lon', 'lat', 'lon_end', 'lat_end', 'snapshots'):=get_coordinate(POLYLINE), by=r]
test[, c('lon', 'lat', 'lon_end', 'lat_end', 'snapshots'):=get_coordinate(POLYLINE), by=r]
train[, POLYLINE:=NULL]
test[, POLYLINE:=NULL]

for (i in 1:nrow(test)) {
      ### Get the distance from each train trip to each test trip
      train[, c('lon2', 'lat2', 'snapshots2'):=test[i, list(lon, lat, snapshots)]]
      train[, d:=get_dist(lon, lat, lon2, lat2)]
      
      ### Get the closest trips to each test trip
      ### Bound below by 10 meters since we use 1/distance^2 as weight
      ### Trips must last as long as 80% of test duration
      ### Trimmed mean of x and y coordinate, assuming flat earth
      ds <- train[snapshots>=snapshots2*.8][order(d)][1:N_trips][!is.na(r), list(lon_end, lat_end)]  
      test[i, c('LONGITUDE', 'LATITUDE'):=ds[, list(mean(lon_end, trim=.1), mean(lat_end, trim=.1))]]    
}

write.csv(test[, list(TRIP_ID, LATITUDE, LONGITUDE)], 'same_start_all_90.csv', row.names=F)
################################################################################
