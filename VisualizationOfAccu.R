setwd("/Users/yangzhang/Downloads/trainset_csv/")
library(readxl)
library( lattice)
library(stringr)



archi1 <- read.csv("archi1.csv")
archi2 <- read.csv("archi2.csv")
archi3 <- read.csv("archi3.csv")
dim(archi3)
#archi3 <- archi3[-1,]

a1 <- seq(1,1000,1)
a2 <- seq(1,100,1)
a3 <- seq(1,500,1)
archi1 <- cbind(a1,archi1)
archi2 <- cbind(a2,archi2)
archi3 <- cbind(a3,archi3)
colnames(archi1) <- c('time','loss','trainAccuracy','valAccuracy')
colnames(archi2) <- c('time','loss','trainAccuracy','valAccuracy')
colnames(archi3) <- c('time','loss','trainAccuracy','valAccuracy')


x <- as.character(archi1[,2])
y <- as.numeric(str_extract_all(x, "\\(?[0-9,.]+\\)?"))
archi1[,2] <- y

x <- as.character(archi1[,3])
y <- as.numeric(str_extract_all(x, "\\(?[0-9,.]+\\)?"))
archi1[,3] <- y

x <- as.character(archi1[,4])
y <- as.numeric(str_extract_all(x, "\\(?[0-9,.]+\\)?"))
archi1[,4] <- y




dev.off()
plot(archi1[,1], archi1[,2], type="o", col="blue", pch="o", lty=1, ylim=c(0.3,1.2),xlab="epochs", ylab="loss/accuracy",main="ResultForCube6x6x6" )
points(archi1[,1], archi1[,3], col="red", pch="*")
lines(archi1[,1], archi1[,3], col="red",lty=2)
points(archi1[,1], archi1[,4], col="black",pch="+")
lines(archi1[,1], archi1[,4], col="black", lty=3)
legend(700,1.3,legend=c("loss","TrainAccu","ValiAccu"), col=c("blue","red","black"),pch=c("o","*","+"),lty=c(1,2,3), ncol=1)


plot(archi2[,1], archi2[,2], type="o", col="blue", pch="o", lty=1, ylim=c(0.3,1.2),xlab="epochs", ylab="loss/accuracy",main="ResultForCube10x10x10" )
points(archi2[,1], archi2[,3], col="red", pch="*")
lines(archi2[,1], archi2[,3], col="red",lty=2)
points(archi2[,1], archi2[,4], col="black",pch="+")
lines(archi2[,1], archi2[,4], col="black", lty=3)
legend(70,1.3,legend=c("loss","TrainAccu","ValiAccu"), col=c("blue","red","black"),pch=c("o","*","+"),lty=c(1,2,3), ncol=1)


plot(archi3[,1], archi3[,2], type="o", col="blue", pch="o", lty=1, ylim=c(0.3,1.4),xlab="epochs", ylab="loss/accuracy",main="ResultForCube20x20x20" )
points(archi3[,1], archi3[,3], col="red", pch="*")
lines(archi3[,1], archi3[,3], col="red",lty=2)
points(archi3[,1], archi3[,4], col="black",pch="+")
lines(archi3[,1], archi3[,4], col="black", lty=3)
legend(300,1.5,legend=c("loss","TrainAccu","ValiAccu"), col=c("blue","red","black"),pch=c("o","*","+"),lty=c(1,2,3), ncol=1)
