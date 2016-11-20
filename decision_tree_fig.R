setwd("C:/Users/ThinkPad User/Desktop")
#my tree 2
library(rpart.plot)
library(rpart)
library(rattle)
#fit <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis)
fit <- rpart(data1[,17] ~ data1[,5]+data1[,6]+data1[,7]+data1[,8]+data1[,9]+data1[,10]+data1[,11]+data1[,12]+data1[,13]+data1[,14]++data1[,15]+data1[,16], data = data1)
fancyRpartPlot(fit)

str(iris)
set.seed(1234)
#从iris数据集中随机抽70%定义为训练数据集，30%为测试数据集
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
library(party)
#建立决策树模型预测花的种类
myFormula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree <- ctree(myFormula, data=trainData)
plot(iris_ctree, type="simple")

#my tree 1
filename<-"training.csv"
data1<-read.table(filename,header=FALSE,sep = ",")
#normalization
#data1<-as.data.frame(scale(data1, center=T,scale=T))
myFormula <- data1[,17] ~ data1[,1]+data1[,2]+data1[,3]+data1[,4]+data1[,5]+data1[,6]+data1[,7]+data1[,8]+data1[,9]+data1[,10]+data1[,11]+data1[,12]+data1[,13]+data1[,14]++data1[,15]+data1[,16]
my_ctree <- ctree(myFormula, data=data1)
#plot(iris_ctree, type="simple")
plot(my_ctree)

library("party")
iris_ctree <- ctree(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=iris)
print(iris_ctree)
plot(iris_ctree)