# survival + random forests for churn
# by pedro.concejerocerezo@telefonica.com
# 8th june  2016
# This is a basic R script intended to be distributed to participants in talk
# so that anyone can reproduce the analyses in their environments (RStudio recommended)

# Aims:
# A reproducible example with public data
# A walkthrough on the capabilities of R library, Random Forests Survival (RFS) by Hemant Ishwaran:
# http://www.ccs.miami.edu/~hishwaran/ishwaran.html

# together with other ones for:
# Visualization to understand survival objects
# How to obtain risk scores from these models
# Computing predictive capability

#######################################
# 1 Setup your R environment 

# Be sure to change your directory!!!!
  
setwd("d:/survival")

# Load (and install if necessary) some required libraries 
# (but see below special requirement about the randomForestSRC library) !!!!

list.of.packages <- c("survival", 
                      "glmnet",
                      "rms",
                      "doParallel",
                      "risksetROC", 
                      "party", 
                      "ROCR",
                      "ggplot2",
                      "survminer",
                      "randomForestSRC",
                      "ggRandomForests")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(survival, quietly = TRUE)
library(glmnet, quietly = TRUE)
library(rms, quietly = TRUE)
library(risksetROC, quietly = TRUE)
library(party, quietly = TRUE)
library(ROCR, quietly = TRUE)
library(ggplot2, quietly = TRUE)
library(survminer, quietly = TRUE)

# doParallel provides multicore functionality on Unix-like systems and snow functionality on Windows
library(doParallel, quietly = TRUE)                    
registerDoParallel(detectCores() - 1 )  ## registerDoMC( detectCores()-1 ) in Linux

detectCores()
options(rf.cores = detectCores() - 1, 
        mc.cores = detectCores() - 1)  ## Cores for parallel processing

# randomForestSRC package allows parallelization but the library binaries are different for Windows and Linux, 
# so you must go to http://www.ccs.miami.edu/~hishwaran/rfsrc.html

#install.packages("http://www.ccs.miami.edu/~hishwaran/rfsrc/randomForestSRC_1.6.0.zip", 
#                 repos = NULL, 
#                 type = "source")
library(randomForestSRC)
                                                                                                                            
#install.packages("ggRandomForests", 
#                 repos = 'http://cran.us.r-project.org') #since you had source before
library(ggRandomForests)


#######################################
# 2 Obtain Data

nm <- read.csv("http://www.sgi.com/tech/mlc/db/churn.names", 
               skip = 4, 
               colClasses = c("character", "NULL"), 
               header = FALSE, 
               sep = ":")[[1]]

dat <- read.csv("http://www.sgi.com/tech/mlc/db/churn.data", 
                header = FALSE, 
                col.names = c(nm, "Churn"),
                colClasses = c("factor",
                               "numeric",
                               "factor",
                               "character",
                               rep("factor", 2),
                               rep("numeric", 14),
                               "factor"))
# test data

test <- read.csv("http://www.sgi.com/tech/mlc/db/churn.test", 
                 header = FALSE, 
                 col.names = c(nm, "Churn"),      # CAVEAT! First C uppercase
                 colClasses = c("factor",
                                "numeric",
                                "factor",
                                "character",
                                rep("factor", 2),
                                rep("numeric", 14),
                                "factor"))


# A quick exploration of data (train)

dim(dat)       # Remember these are training data
summary(dat)   # Note the data types

length(unique(dat$phone.number))

hist(dat$account.length)       # This is essential to prepare the survival object
table(dat$Churn)/nrow(dat)*100 # Percent of churners. Note it is a factor


# And about the test set. You have exactly 1667 rows, exactly half of the training set.

summary(test);dim(test)

#######################################
# 3. EXPLORE SURVIVAL BASICS

# Survival object requires a numeric (0/1) target (!!)

dat$Churn <- as.numeric(dat$Churn) - 1

s <- with(dat, 
          Surv(account.length, Churn))

head(s, n = 100)
class(s)

## Kaplan-Meier estimator. The "log-log" confidence interval is preferred.
km.as.one <- survfit(s ~ international.plan, #or area.code,
                     data = dat, 
                     conf.type = "log-log")

## Show object
km.as.one

# Basic plot
plot(km.as.one,
     xlab = "Tenure- months",
     ylab = "Prob. survival",
     ylim = c(0.2, 1))
title("Kaplan-Meier per international plan")

# 
ggsurvplot(km.as.one,
           break.time.by = 12,
           color = "red",
           surv.scale = "percent",
           xlab = "Tenure- months",
           ylab = "% survived",
           ylim = c(0.2, 1),
           censor = T,
           legend = "top",
           risk.table = T,
           ggtheme = theme_gray())

#######################################
# 4 Let's go for rsfsrc

# rfsrc requires all data to be either numeric or factors
# Filter out some irrelevant factors and the id number

dat$phone.number <- NULL
dat$state <- NULL
dat$area.code <- NULL

# these are relevant factors but are converted into dummies -worked better with v1.6 library

dat$international.plan <- as.numeric(dat$international.plan) - 1
dat$voice.mail.plan <- as.numeric(dat$voice.mail.plan) - 1

# Final training data

summary(dat)


# Let's try a simple RF model with 50 trees and nsplit 2.

out.rsf.1 <- rfsrc(Surv(account.length, Churn) ~ . , 
                   data = dat,
                   ntree = 50, 
                   nsplit = 1,
                   forest = T,   # required to use vimp functions
                   tree.err = T, # required for plotting tree error
                   importance = T)

out.rsf.1

# this is a complex object containing all the info. about the input and output of model
str(out.rsf.1)

plot(out.rsf.1)

# To obtain variable importance we can use standard vimp function (requires forest = T when growing the forest)
# You have several ways of computing importance see manual
# The default is Breiman-Cutler permutation VIMP

vimp(out.rsf.1)

var.importance <- (vimp(out.rsf.1)$importance)

print(sort(var.importance, decreasing = T))

# You can plot results above using standard barplot or use gg_vimp function

plot(gg_vimp(out.rsf.1))



## Predictive capability in the training set
# we use OOB data
# and in particular predicted.oob vector 
# contains predicted survival times for all train data

summary(out.rsf.1$predicted.oob)

# In RSF, error rate is defined as = 1 – C-index
# we use Hmisc::rcorr.cens rank correlation for censored data
# see ?rcorr.cens

## C-index
rcorr.cens(out.rsf.1$predicted.oob, 
           Surv(dat$account.length, dat$Churn))["C Index"]

## Inverting this amount so that the higher the better 
## OOB Error = 1 – C-index
rcorr.cens(-out.rsf.1$predicted.oob, 
           Surv(dat$account.length, dat$Churn))["C Index"]


# This is similar to the error rate computed by the algorithm 
# and reported in the err.rate vector -for all numbers of trees

out.rsf.1$err.rate

# and for the specified number of trees it is a single number:
out.rsf.1$err.rate[ out.rsf.1$ntree ]



#######################################
# 5 Improving-optimizing RFS

out.rsf.2 <- rfsrc( Surv(account.length, Churn) ~ . , 
                    data = dat, 
                    ntree = 70, 
                    importance = "none", 
                    tree.err = T,
                    nsplit = 2)
out.rsf.2
plot(out.rsf.2)

out.rsf.3 <- rfsrc( Surv(account.length, Churn) ~ . , 
                    data = dat, 
                    ntree = 200, 
                    importance = "none", 
                    tree.err = T,
                    nsplit = 1)
out.rsf.3
plot(out.rsf.3)

out.rsf.4 <- rfsrc( Surv(account.length, Churn) ~ . , 
                    data = dat, 
                    ntree = 500, 
                    importance = "none", 
                    tree.err = T,
                    nsplit = 1)
out.rsf.4

# As suspectetd, not worth increasing number of trees
# nsplit does not seem to affect

# we will use out.rsf.3

#######################################
# 6 Predictive ability with test data: C Index

# First of all, remember to make same mods as we did to the training to test set!!!

test$phone.number <- NULL
test$state <- NULL
test$area.code <- NULL

test$international.plan <- as.numeric(test$international.plan) - 1
test$voice.mail.plan <- as.numeric(test$voice.mail.plan) - 1
test$Churn <- as.numeric(test$Churn) - 1

summary(test)


# we predict over test set with predict function and newdata = test

pred.test.fin = predict( out.rsf.3, 
                         newdata = test, 
                         importance = "none" )

# Computing C index as we did with train data (inverted so we report 1-C)

rcorr.cens(-pred.test.fin$predicted , 
           Surv(test$account.length, test$Churn))["C Index"]


#######################################
# 7 Estimates of survival time and risk in test data

# Remember we do not have a single risk estimate per individual as with classification models
# We have a risk estimate *per* time moment and an estimated survival time (the lower the worse)

# In pred.test.fin object we have the following vectors
# $time.interest contains survival times when events have happened (of course right-censored)
summary(pred.test.fin$time.interest)

# $predicted has the predictions -note you have an .oob object but it is empty
summary(pred.test.fin$predicted)

# $survival contains survival estimates for each individual *at each moment in time*
dim(pred.test.fin$survival)

# We can explore survival estimates at specific moments in time
summary(pred.test.fin$survival[, 12])
summary(pred.test.fin$survival[, 24])

# And $chf contains accumulated risk (kinda sum of risks) per individual
# CAVEAT THIS IS NOT A PROBABILITY. The higher the worse -sum of risks
# and as survival estimates you have one estimate per moment in time
summary(pred.test.fin$chf)

summary(pred.test.fin$chf[, 12])
summary(pred.test.fin$chf[, 24])


#######################################
# 8 Survival ROC at specific moments in time using risksetROC
# A key concept in all survival modelling is prediction along time

## Survival ROC at t=10
w.ROC = risksetROC(Stime = test$account.length, 
                   status = test$Churn,
                   marker = pred.test.fin$predicted, 
                   predict.time = 10, 
                   method = "Cox",
                   main = "Test Survival ROC Curve at t=10", 
                   lwd = 3, 
                   col = "red" )

w.ROC$AUC

## Survival ROC at t=24

w.ROC = risksetROC(Stime = test$account.length, 
                   status = test$Churn,
                   marker = pred.test.fin$predicted, 
                   predict.time = 24, 
                   method = "Cox",
                   main = "Test Survival ROC Curve at t=10", 
                   lwd = 3, 
                   col = "red" )

w.ROC$AUC

# For risksetROC to compute AUC along an interval you use risksetAUC using tmax (maximum time). 
# You get a very nice plot of AUC across time. 

w.ROC = risksetAUC(Stime = test$account.length, 
                   status = test$Churn,
                   marker = pred.test.fin$predicted, 
                   tmax = 190, 
                   method = "Cox",
                   main = paste("OOB Survival ROC Curve at t=190, test data"), 
                   lwd = 3, 
                   col = "red" )

