# Data from http://stackoverflow.com/questions/27080207/survival-analysis-for-telecom-churn-using-r
# Churn data (artificial based on claims similar to real world)

# FANTASTIC PPT AT 
# 

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

summary(dat);dim(dat)

# 3333 unique customer id's (phone numbers)
length(unique(dat$phone.number))

# account.length is the age (time dimension)
# seems to be months
hist(dat$account.length)

# 15% is quite a high churn rate but consider we have >10 years span
table(dat$Churn)/nrow(dat)*100

# test data

test <- read.csv("http://www.sgi.com/tech/mlc/db/churn.test", 
                header = FALSE, 
                col.names = c(nm, "Churn"),
                colClasses = c("factor",
                               "numeric",
                               "factor",
                               "character",
                               rep("factor", 2),
                               rep("numeric", 14),
                               "factor"))

summary(test);dim(test)


library(survival)
setwd("d:/survival")
source("ggsurv.r")

s <- with(dat, 
          Surv(account.length, as.numeric(Churn)))

head(s, n = 100)
class(s)

## Kaplan-Meier estimator. The "log-log" confidence interval is preferred.
km.as.one <- survfit(s ~ 1, data = dat, conf.type = "log-log")

## Show object
km.as.one
plot(km.as.one,
     xlab = "Tenure- months",
     ylab = "Prob. survival")
title("Kaplan-Meier all data")

## Kaplan-Meier estimator per group
km.area.code <- survfit(s ~ area.code, data = dat, conf.type = "log-log")

## Show object
km.area.code
plot(km.area.code,
     xlab = "Tenure- months",
     ylab = "Prob. survival")
title("Kaplan-Meier per area.code")


#============================================================
# Code based on rattle
# 1. Conditional inference tree. 
# Build a conditional tree using the party package.

require(party, quietly = TRUE)

# Build a ctree model.

rpart <- ctree(Churn ~ ., 
               data = dat[, c(2, 3,
                              5:21)])

# Generate summary of the ctree model.

print(rpart)
plot(rpart) # Cannot see anything, this requires plot to a file

png(filename = "rpart_tree_churn_survival.png",
    width = 3000,
    height = 1500)
plot(rpart) 
dev.off()

# Let's see some survival curves based on relevant variables identified

library(rms)

## Kaplan-Meier estimator per group
# Introducing npsurv (nonparametric surv function) for using rms for plotting
km.int.calls <- npsurv(formula = Surv(account.length, as.numeric(Churn)) ~ international.plan,
                       data = dat)


## Show object
km.int.calls
survplot(km.int.calls,
         xlab = "Tenure- months",
         ylab = "Prob. survival")
title("Kaplan-Meier per international calls")

#########
# Evaluate rpart model performance. 
# ROC Curve: requires the ROCR package.

library(ROCR)

# ROC Curve: requires the ggplot2 package.

require(ggplot2, quietly = TRUE)

# Generate an ROC Curve for the rpart model on dat [test].
pr <- predict(rpart,
              newdata = test[, c(2, 3,
                                 5:21)],
              type = c("prob"))

kk <- sapply(pr, "[[", 2)

pred <- prediction(kk,
                   test$Churn)

pe <- performance(pred, "tpr", "fpr")
au <- performance(pred, "auc")@y.values[[1]]
pd <- data.frame(fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))
p <- ggplot(pd, aes(x=fpr, y=tpr))
p <- p + geom_line(colour="red")
p <- p + xlab("False Positive Rate") + ylab("True Positive Rate")
p <- p + ggtitle("ROC Curve Decision Tree dat [test] Churn")
p <- p + theme(plot.title=element_text(size=10))
p <- p + geom_line(data=data.frame(), aes(x=c(0,1), y=c(0,1)), colour="grey")
p <- p + annotate("text", x=0.50, y=0.00, hjust=0, vjust=0, size=5,
                  label=paste("AUC =", round(au, 2)))
print(p)

# Calculate the area under the curve for the plot.
performance(pred, "auc")


#============================================================
# Cox regression survival model
# Build a simple Survival model.

#################################################################

model <- coxph(s ~ international.plan, 
               data = dat[, c(2, 3,
                              5:21)])
summary(model)
plot(survfit(model))

# A quick and dirty example of finding significant factors for churn

model <- coxph(s ~ international.plan + strata(number.customer.service.calls <= 3), 
               data = dat[, c(2, 3,
                              5:21)])
summary(model)
plot(survfit(model), 
     col = c("blue", "red"))

ggsurv(survfit(model))


# Check the PH assumption - log-log plot

f <- survfit(Surv(account.length, as.numeric(Churn)) ~ international.plan,
             data = dat[, c(2, 3,
                            5:21)])

plot(f,
     fun = "cloglog",
     col = c("blue", "red"),
     xlab = "log(t)",
     ylab = "log(Hazard)")

legend(legend = c("no int.calls plan", "yes int.calls.plan"),
       x = 1,
       pch = 3,
       col = c("blue", "red"))

title("Log-log plot to test PH assumption- predictor international plan")


# Extended Cox-model
# using library rms

library(rms)

ddist <- datadist(dat[, c(2, 3,
                          5:21)])
options(datadist = 'ddist')
model <- rms::cph(s ~ international.plan + 
                      rcs(number.customer.service.calls, 3) +
                      voice.mail.plan, 
                  data = dat[, c(2, 3,
                              5:21)],
                  x = T,
                  y = T)
summary(model)

plot(summary(model, 
             number.customer.service.calls = c(0, 3)),
     log = T,
     col = c("orange", "gold", "blue"))


#################################################################
# extended cox - full model 


fullmodel <- rms::cph(s ~  international.plan + 
                           voice.mail.plan + 
#                           number.vmail.messages + 
#                           total.day.minutes + 
#                           total.day.calls + 
#                           total.day.charge + 
#                           total.eve.minutes + 
#                           total.eve.calls + 
#                           total.eve.charge + 
#                           total.night.minutes + 
#                           total.night.calls + 
#                           total.night.charge + 
#                           total.intl.minutes + 
                           total.intl.calls + 
#                           total.intl.charge + 
                           number.customer.service.calls,
                      data = dat[, c(2, 3,
                                     5:21)],
                      surv = T
)

# Print the results of the modelling.

summary(fullmodel)


plot(summary(fullmodel), 
     log = T,
     col = c("orange", "gold", "blue"))


##########################‼
# Predicting survival probabilities 

library(pec)

kk <- predictSurvProb(fullmodel, 
                      newdata = test[, c("international.plan",
                                 "voice.mail.plan",
                                 "total.intl.calls",
                                 "number.customer.service.calls")],
                      times = max(test$account.length))

# with predictSurvProb you can predict at specific moments in time!!!!


# Generate an ROC Curve for the rpart model on dat [test].

pred <- prediction((1-kk),     # to accomodate we have survival NOT DROP probabilities
                   test$Churn)

pe <- performance(pred, "tpr", "fpr")
au <- performance(pred, "auc")@y.values[[1]]
pd <- data.frame(fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))
p <- ggplot(pd, aes(x=fpr, y=tpr))
p <- p + geom_line(colour="red")
p <- p + xlab("False Positive Rate") + ylab("True Positive Rate")
p <- p + ggtitle("ROC Curve Decision Tree dat [test] Churn")
p <- p + theme(plot.title=element_text(size=10))
p <- p + geom_line(data=data.frame(), aes(x=c(0,1), y=c(0,1)), colour="grey")
p <- p + annotate("text", x=0.50, y=0.00, hjust=0, vjust=0, size=5,
                  label=paste("AUC =", round(au, 2)))
print(p)

# Calculate the area under the curve for the plot.
performance(pred, "auc")


