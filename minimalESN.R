# install.packages('R6')
library(R6) # OOP library
set.seed(42)

# ESN Class
ESNet <- R6Class("ESNet",
               public = list(
                 in.size = 1, # input size
                 out.size = 1, # output size
                 res.size = 500, # reservoir size
                 leaking.rate = 0.5,
                 sparsity = 0.6, # % of zeros in the rezervoir
                 spectral.radius = 0.9,
                 reg = 1e-8,
                 Yt = matrix(),
                 x = matrix(),
                 Win = matrix(),
                 Wout = matrix(),
                 W = matrix(),
                 data = matrix(),
                 
                 # Initialization method
                 initialize = function(in.size = 1,
                                       out.size = 1,
                                       res.size = 500,
                                       leaking.rate = 0.5,
                                       sparsity = 0.6,
                                       spectral.radius = 0.9,
                                       reg = 1e-8) {
                   self$in.size <- in.size
                   self$out.size <- out.size
                   self$res.size <- res.size
                   self$leaking.rate <- leaking.rate
                   self$sparsity <- sparsity
                   self$spectral.radius <- spectral.radius
                   self$reg <- reg
                 },
                 
                 # Fitting function
                 fit = function(data) {
                   
                   Win <- matrix(runif(self$res.size*(1+self$in.size),-0.5,0.5), self$res.size)
                   
                   W <- runif(self$res.size*self$res.size,-0.5,0.5)
                   n_zeros <- round(self$res.size*self$res.size*self$sparsity, digits=0)
                   zero_index <- sample(self$res.size*self$res.size)[1:n_zeros]
                   W[zero_index] <- 0 
                   W <- matrix(W, self$res.size)
                   rhoW <- abs(eigen(W, only.values=TRUE)$values[1])
                   W <- W * self$spectral.radius / rhoW
                   
                   
                   X <- matrix(0, 1+self$in.size+self$res.size, length(data)-1)
                   
                   Yt <- matrix(data[2:length(data)], 1)
                   x <- rep(0,self$res.size)
                   a <- self$leaking.rate
                   
                   for(t in 1:(length(data)-1)){
                     u <- data[t]
                     x <- (1-a)*x + a*tanh( Win %*% rbind(1,u) + W %*% x )
                     X[, t] <- rbind(1,u,x)
                   }
                   
                   # train the output
                   reg <- self$reg  # regularization coefficient
                   X_T <- t(X)
                   Wout <- Yt %*% X_T %*% solve( X %*% X_T + reg*diag(1+self$in.size+self$res.size ), tol = 1e-18 )
                   
                   #cat("Treinamento finalizado")
                   
                   self$Yt <- Yt
                   self$x <- x
                   self$Win <- Win
                   self$Wout <- Wout
                   self$W <- W
                   self$data <- data
                 },
                 
                 
                 predict = function(steps, generative=TRUE) {
                   Y <- matrix(0, self$out.size, steps)
                   u <- self$Yt[length(self$Yt)]
                   x <- self$x

                   for (t in 1:steps) {
                     x <- (1-self$leaking.rate)*x + self$leaking.rate*tanh( self$Win %*% rbind(1,u) + self$W %*% x )
                     y <- self$Wout %*% rbind(1,u,x)
                     Y[,t] <- y
                     if(generative) {
                       # generative mode:
                       u <- y
                     } else {
                       # predictive mode:
                       u = self$data[t+1]
                     }
                     
                     #
                     
                   }
                   return(t(Y))
                 }
               )
)

## Functions to use in R 'style'

# Create ESN function
ESN <- function(in.size = 1, # tamanho da entrada
                out.size = 1, # tamanho da saída
                res.size = 500, # tamanho do reservatório
                leaking.rate = 0.5,
                sparsity = 0.6,
                spectral.radius = 0.9,
                reg = 1e-8) {
  
  return(ESNet$new(in.size = in.size,
                 out.size = out.size,
                 res.size = res.size,
                 leaking.rate = leaking.rate,
                 sparsity = sparsity,
                 spectral.radius = spectral.radius,
                 reg = reg))
}

# Train ESN function
train_esn <- function(esn, data){
  esn$fit(data)
  return(esn)
}

# Generate predictions function
predict_esn <- function(esn, steps, generative = TRUE){
  return(esn$predict(steps, generative))
}





