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
                      "survminer")

library(party)
library(ROCR)
library(ggplot2)
library(survminer)


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

install.packages("http://www.ccs.miami.edu/~hishwaran/rfsrc/randomForestSRC_1.6.0.zip", 
                 repos = NULL, 
                 type = "source")
library(randomForestSRC)
                                                                                                                            
install.packages("ggRandomForests", 
                 repos = 'http://cran.us.r-project.org') #since you had source before
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
                   nsplit = 2)
out.rsf.1


# The $importance object contains variables importance, in same order as in input dataset. 
# We sort it out to show a ranking and use ggRandomForestses to plot this object. 

imp.rsf.1 <- sort(out.rsf.1$importance, 
                  decreasing = T)
imp.rsf.1

plot(gg_vimp(out.rsf.1))


## Predictions in RSF
# These predictions can be based on all trees or only on the OOB sample.

# We have the following predictions
# predicted
# predicted.oob
# survival
# survival.oob

summary(out.rsf.1$survival) # all values between 0 and 1
summary(out.rsf.1$survival.oob) # same but different values
summary(out.rsf.1$predicted) # clearly a duration - between 0 and 173
summary(out.rsf.1$predicted.oob) # same but different values between 0 and 165

# We must make a transform to make these equivalent to a risk score, 
# so that higher values correspond to observations with higher observed risk, lower survival. 


# In RSF, error rate is defined as = 1 – C-index

## OOB Error = 1 – C-index
rcorr.cens(out.rsf.1$predicted.oob, 
           Surv(dat$account.length, dat$Churn))["C Index"]

err.rate.rsf = out.rsf.1$err.rate[ out.rsf.1$ntree ]
err.rate.rsf

## Inverting this amount so that the higher the better  
rcorr.cens(-out.rsf.1$predicted.oob, 
           Surv(dat$account.length, dat$Churn))["C Index"]


#######################################
# 5 Towards an optimal RFS

out.rsf.3 <- rfsrc( Surv(account.length, Churn) ~ . , 
                    data = dat, 
                    ntree = 200, 
                    importance = "none", 
                    nsplit = 1)
out.rsf.3

plot(gg_error(out.rsf.3))


## Predictive ability applied to test data: C Index

# First of all, remember to make same mods as we did to the training to test set!!!

test$phone.number <- NULL
test$state <- NULL
test$area.code <- NULL

test$international.plan <- as.numeric(test$international.plan) - 1
test$voice.mail.plan <- as.numeric(test$voice.mail.plan) - 1
test$Churn <- as.numeric(test$Churn) - 1

summary(test)


pred.test.fin = predict( out.rsf.3, 
                         newdata = test, 
                         importance = "none" )

rcorr.cens(-pred.test.fin$predicted , 
           Surv(test$account.length, test$Churn))["C Index"]


w.ROC = risksetROC(Stime = dat$account.length,  
                   status = dat$Churn, 
                   marker = out.rsf.3$predicted.oob, 
                   predict.time = median(dat$account.length), 
                   method = "Cox", 
                   main = paste("OOB Survival ROC Curve at t=", 
                                median(dat$account.length)), 
                   lwd = 3, 
                   col = "red" )

w.ROC$AUC


# Let's see prediction at 12 months

w.ROC = risksetROC(Stime = dat$account.length,  
                   status = dat$Churn, 
                   marker = out.rsf.3$predicted.oob, 
                   predict.time = 12, 
                   method = "Cox", 
                   main = paste("OOB Survival ROC Curve at t=", 
                                median(dat$account.length)), 
                   lwd = 3, 
                   col = "red" )

w.ROC$AUC

# For risksetROC to compute AUC along an interval you use risksetAUC using tmax (maximum time). 
# You get a very nice plot of AUC across time. 

# *******************This is still OOB samples.*****************************


w.ROC = risksetAUC(Stime = dat$account.length,  
                   status = dat$Churn, 
                   marker = out.rsf.3$predicted.oob,
                   tmax = 250)

# **************Let's do the same for test data.********************

w.ROC = risksetAUC(Stime = test$account.length,  
                   status = test$Churn, 
                   marker = pred.test.fin$predicted, 
                   tmax = 190, 
                   method = "Cox",
                   main = paste("OOB Survival ROC Curve at t=190"), 
                   lwd = 3, 
                   col = "red" )

w.ROC$AUC
