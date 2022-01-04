#Limpieza y análisis del dataset Titanic

#Librerías
library(readr)
library(data.table)
library(tidyverse)
library(nortest)
library(PerformanceAnalytics)
#library(glmnet)
library(caret)
library(dplyr)
library(ROCR)
library(ggplot2)
library(gridExtra)
#library(vcd)

##################################
# 1. Descripción del dataset ####
#################################

# El dataset Titanic elegido de Kaggle, cuenta con 3 archivos de datos en formato csv. 
# Para esta practica elegiremos el dataset de train.csv pues es el que cuenta con 
# un mayor numero de valores 

#Cargamos el dataset de train
titanic <- read.csv('/Users/juanmatillavaras/Desktop/Master\ Data\ Science/Tipología/PRAC2/titanic/train.csv')
#titanic <- read.csv('/Users/alber/Dropbox/Tipologia_UOC/PRAC2/titanic/train.csv')

#Veamos la cabecera
head(titanic)

#Veamos un resumen de nuestros datos
str(titanic)
summary(titanic)

################################################
# 2. Integracion y seleccion de los datos ######
################################################

titanic$PassengerId <- NULL
titanic$Name <- NULL
titanic$Ticket <- NULL

##################################
# 3. Limpieza de los datos ######
#################################

# 3.1. Valores perdidos

# Unificamos por si acaso todos los posibles valores perdidos
unify_null <- function(x){
  na_index<-(is.na(x) | x=="null" | x=="NULL" | x=='\\N' | x=='\\n' | x=="" | x=="?" | x=='NA');
  x[na_index]<-NA;
  return(x)
}

#Aplicamos la función al dataset
titanic <- data.table(apply(titanic, 2, unify_null))

#Veamos cuantos valores perdidos tiene cada variable
sapply(titanic, function(x){sum(is.na(x))})
#Porcentaje de valores perdidos
sapply(titanic, function(x){100*sum(is.na(x))/length(x)})
#Vemos que tenemos casi un 20% de valores perdidos en Age, un 77% en Cabin y un 
# 0.22 en Embarked. El resto de variables no tienen valores perdidos.

#Eliminamos la variable Cabin ya que tiene demasiados valores perdidos.
titanic$Cabin <- NULL

#A continuación vamos a reemplazar los valores perdidos.
#Veamos los datos estadísticos de edad para decidir como reemplazamos los valores perdidos.
titanic$Age <- as.integer(titanic$Age)
summary(titanic$Age, na.rm = TRUE)
hist(titanic$Age)
#Viendo estos datos podemos reemplazar por la media
titanic$Age[is.na(titanic$Age)] <- mean(titanic$Age, na.rm = TRUE)
summary(titanic$Age)
hist(titanic$Age)
#Los datos han cambiado ligeramente aunque la distribución no se ha visto demasiado alterada

#Seguimos el mismo proceso para la variable del embarque.
titanic$Embarked <- as.factor(titanic$Embarked)
summary(titanic$Embarked, na.rm = TRUE)
#En este caso vamos a reemplazar los dos valores perdidos por el valor que más aparece: 'S'
titanic$Embarked[is.na(titanic$Embarked)] <- 'S'

#Revisamos de nuevo los valores perdidos del dataset
sapply(titanic, function(x){sum(is.na(x))})
#Como podemos ver, ya no tenemos valores perdidos.

# 3.2. Outliers

#Lo primero que vamos a hacer es asignar correctamente los tipos de cada variable.
titanic$Age <- as.numeric(titanic$Age)
titanic$Fare <- as.numeric(titanic$Fare)
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$Sex <- as.factor(titanic$Sex)
titanic$SibSp <- as.factor(titanic$SibSp)
titanic$Parch <- as.factor(titanic$Parch)
titanic$Embarked <- as.factor(titanic$Embarked)
titanic$Survived <- as.factor(titanic$Survived)

#A continuación hacemos unas sencillas visualizaciones para detectar posibles outliers.
plot(titanic$Survived)
plot(titanic$Pclass)
plot(titanic$Sex)
boxplot(titanic$Age)
plot(titanic$SibSp)
plot(titanic$Parch)
boxplot(titanic$Fare)
plot(titanic$Embarked)

#Parece que en el caso de la variable Fare tenemos un problema de outliers.
summary(titanic$Fare)
fare_plot <- boxplot(titanic$Fare)
fare_plot$out

#Como podemos ver, llamando alcomando out parece que tenemos una gran cantidad de 
# outliers en esta variable, así que podemos eliminarlos
titanic <- titanic[!(titanic$Fare %in% fare_plot$out),]
boxplot(titanic$Fare)
summary(titanic$Fare)
#Podemos ver como el boxplot ha cambiado así como los datos estadísticos.

##################################
# 4. Análisis de los datos ######
#################################

# 4.1. Selección de los grupos

#En este caso, analizaremos los datos para los pasajeros de clase baja
titanic$Clase_baja <- ifelse(titanic$Pclass == 3, TRUE, FALSE)

# 4.2. Comprobación de normalidad y homocedasticidad

#Test de normalidad

shapiro.test(titanic$Age)
ks.test(titanic$Age, pnorm, mean(titanic$Age), sd(titanic$Age))

shapiro.test(titanic$Fare)
ks.test(titanic$Fare, pnorm, mean(titanic$Fare), sd(titanic$Fare))

#Test de homocedasticidad

fligner.test(Age ~  Survived, data = titanic)
fligner.test(Fare ~ Survived, data = titanic)

# 4.3. Pruebas estadísticas

# 4.3.1. Correlaciones

correlaciones <- titanic[, c(4,7)]
chart.Correlation(correlaciones, histogram = TRUE, method = "pearson")

# 4.3.2. Contraste de hipótesis

# Precio del billete de los que sobreviven
sobrevive <- titanic %>% filter(Survived == 1) %>% pull(Fare)
# Precio del billete de los que no sobreviven
nosobrevive <- titanic %>% filter(Survived == 0) %>% pull(Fare)
# El test de Wilcoxon 
wilcox.test(x=sobrevive,y=nosobrevive, paired = F)
mean(sobrevive)
mean(nosobrevive)
mean(sobrevive) - mean(nosobrevive)

# 4.3.3. Regresión logística.

# Creamos indices para dividir la base en entrenamiento y test con una misma 
# relacion de la variable objetivo entre ellas y una division de 80-20
train_index <- createDataPartition(y = titanic$Survived, p = 0.8, list = FALSE, times = 1)
dat_train <- titanic[train_index, ]
dat_test  <- titanic[-train_index, ]
train_index <- sample(1:nrow(titanic), 0.8*nrow(titanic))  
dat_train <- titanic[train_index, ] 
dat_test  <- titanic[-train_index, ]

#Split Label
train_label <- dat_train$Survived
test_label <- dat_test$Survived

# Creamos el modelo logic
modelo1<-glm(Survived ~ Clase_baja +  Age + Fare, data=dat_train, na.action = "na.omit", family = "binomial")
# Vemos los resultados
summary(modelo1)

#Predicción
pred_logistic <- predict(object = modelo1, newdata =  dat_test, type = "response")
pred_label <- as.factor(ifelse(pred_logistic < 0.6, 0, 1)) 
dat_test$Survived <- as.factor(dat_test$Survived)

#Evaluación
Log_eval <- confusionMatrix(data = pred_label, reference = test_label, positive = "1")
Log_eval

# 5. Representación de los resultados.

boxplot(dat_train$Age~dat_train$Survived, ylab = 'Age', xlab = 'Survived')
boxplot(dat_train$Fare~dat_train$Survived, ylab = 'Fare', xlab = 'Survived')

ggplot(dat_train,aes(x=Survived,fill=Pclass))+
  geom_bar(position = "fill")

# Export del csv resultante

write_csv2(titanic, 'titanic_modificado.csv')

