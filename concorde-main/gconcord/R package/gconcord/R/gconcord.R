#' Graphical CONvex CORrelation selection methoD (gconcord)
#'
#' Estimates a sparse inverse covariance matrix from a convex
#' pseudo-likelihood function with lasso L1 penalty or an elastic net penalty.
#'
#' Coordinate-wise descent algorithm for gconcord is discussed in https://arxiv.org/pdf/1307.5381.pdf, both ISTA
#' and FISTA are discussed in https://arxiv.org/abs/1409.3768.
#'
#'
#' @param data a numerical matrix or a data frame with n observations (rows) and
#'        p variables (columns). \code{NULL} is set only if data are not available and cov will be used.
#' @param S optional, a symmetric numeric matrix. Sample covariance matrix is only required
#'        if data are not available.
#' @param lambda1 a non-negative scalar L-1 penalty parameter, or a p-by-p symmetric penalty matrix.
#' @param lambda2 a non-negative scalar for L-2 penalty parameter. The default is 0.
#' @param method a character string indicating which optimization method is to be used.
#'        It can be one of "\code{coordinatewise}", "\code{ista}", or "\code{fista}". The default is
#'        "\code{coordinatewise}".
#' @param tol a numeric number for convergence threshold in optimization method. The default is \code{1e-05}.
#' @param maxit an integer for the maximum number of iterations before termination in each optimization method.
#'        The default is 100.
#' @param steptype an integer determining the step type for \code{ista} and \code{fista}. Three available choices are:
#' \itemize{
#' \item \code{0}: initial step size is heuristic of Barzilai-Borwein. Please refere to \url{https://arxiv.org/pdf/1409.3768.pdf};
#' \item \code{1}: initial step size is 1;
#' \item \code{2}: initial step size is the feasible step size found from previous iteration.
#' }
#' The default value is \code{0}.
#'
#' @return A list object containing three variables:
#' \itemize{
#' \item\code{omega}: the computed optimal sparse inverse covariance estimator;
#' \item\code{nitr}: the total number of iterations in the computation;
#' \item\code{maxdiff}: a numeric vector presenting the covergence measures for all iterations.
#' }
#'
#' @examples
#' # generate random data
#' p = 4
#' n = 100
#' sigma = matrix(1,nrow = p,ncol = p) * 0.5 + diag(1,p,p) * 0.5
#' set.seed(1)
#' library(MASS);data <- round(mvrnorm(n, mu = rep(0,p), sigma),2)
#'
#' # scalar penalty parameters
#' omega <- gconcord(data, lambda1 = 0.36, lambda2 = 0.1,
#'                   method = "coordinatewise")
#' omega$omega    # the estimated sparse inverse covariance matrix
#' omega$nitr     # the number of iterations in the algorithm
#' omega$maxdiff  # the vector of the convergence measures
#'
#'
#' @useDynLib gconcord
#' @export
gconcord <- function(data,
                     S,
                     lambda1,
                     lambda2 = 0,
                     method = "coordinatewise",
                     tol=1e-5,
                     maxit=100,
                     steptype=0){

  ## obtain sample covariance matrix
  try( if( missing(data) & missing(S) ) stop("Need at least one input for data or cov argument!") )
  if( !missing(data) ){
    std = apply(data, 2, sd)
    datascale = scale( data, center = TRUE, scale = TRUE )
    S = t(datascale) %*% datascale / (nrow(data) - 1)
  }else{
    std = sqrt(diag(S))
    S = diag(1/std) %*% S %*% diag(1/std)
  }
  p = nrow(S)
  nitr = 0
  X0 = diag(1, p)  ## initial estimate of precision matrix

  res = check.penalty( lambda1, lambda2, p ) ## obtain penalty parameters
  lam1 = res$lam1
  lam2 = res$lam2

  if( method == "ista" ){

    out <- .Call("_gconcord_ccista_conv",
                 PACKAGE = 'gconcord',
                 S,
                 as(X0, 'dgCMatrix'),
                 lam1,
                 as.double(lam2),
                 as.double(tol),
                 # as.integer(nitr),
                 as.integer(maxit),
                 as.integer(steptype)
                 )

  }else if( method == "fista" ){

    out <- .Call("_gconcord_ccfista_conv",
                 PACKAGE = 'gconcord',
                 S,
                 as(X0, 'dgCMatrix'),
                 lam1,
                 as.double(lam2),
                 as.double(tol),
                 # as.integer(nitr),
                 as.integer(maxit),
                 as.integer(steptype)
                 )

  }else if( method == "coordinatewise" ){

    out <- .Call("_gconcord_ccorig_conv",
                 PACKAGE = 'gconcord',
                 S,
                 X0,
                 lam1,
                 as.double(lam2),
                 as.double(tol),
                 # as.integer(nitr),
                 as.integer(maxit)
    )
  }

  # output <- list(omega = diag(1/std) %*% matrix(out$out, p, p) %*% diag(1/std),
  #                nitr = out$nitr,
  #                maxdiff = out$maxdiff[1:out$nitr])
  output <- diag(1/std) %*% matrix(out$out, p, p) %*% diag(1/std)

  output

}

#'
#' Cross Validation for the Tuning Parameters in Graphical CONCORD Methods
#'
#' For a given cost function, it estimates the optimal choice of the tuning parameters
#' for L1 penalty and Frobenius norm penalty in the Graphical CONCORD
#' optimization problem. This function can be used only if data are available.
#'
#' @param data a numerical matrix or a data frame with n observations (rows) and p variables (columns).
#' @param rand optional, an integer vector of the length the number of observations (rows) in dataset,
#'        assiging observations to different groups for cross validation. The default is that the function would
#'        automatically create the integer vector under the argument \code{K}.
#' @param FUN optional, a user-defined cost function used in cross validation. It should contain two arguments:
#' \itemize{
#'        \item \code{param}: the estimated parameter, a matrix input of the inverse covariance matrix estimation.
#'        \item \code{data}: the data used in the cost function.
#'        }
#'        If \code{FUN} is messing, predictive risk function (see \code{\link{pred.risk}} for reference) will be used.
#'        That is, \code{FUN = pred.risk} by default.
#' @param lam1.vec optional, a numerical vector containing candidate values for L1 penalty parameter, or a p-by-p penalty matrix. In the
#'        latter case, no cross validation will be conducted for L1 penalty parameter.
#' @param lam2.vec optional, a numerical vector containing candidate values for Frobenius norm penalty parameter.
#' @param K optional, the number of folds in cross validation. Default value is \code{3}.
#' @param method optional, a character string indicating which optimization method is used. See \code{\link{gconcord}} for reference.
#' @param tol optional, a numeric number for convergence threshold in optimization method. The default is \code{1e-05}. See \code{\link{gconcord}}.
#' @param maxit optional, an integer for the maximum number of iterations before termination in each optimization method.
#'        The default is 100. See \code{\link{gconcord}}.
#' @param steptype optional, an integer determining the step type in \code{ista} and \code{fista}. See \code{\link{gconcord}}.
#' @return Function \code{cv.gconcord} returns a list object containing 4 variables:
#' \itemize{
#'   \item{\code{val.error}}{: a validation error matrix for each
#'        candidate choice of lambda_1 and lambda_2. The row names and columns names are values of candidate choice of lambda_1
#'        and lambda_2, respectively. If lambda_1 is a penalty matrix, then the row name is denoted as -1.}
#'   \item{\code{val.error.quantile}}{: the quantile values in matrix for a validation error matrix \code{val.error}.}
#'   \item{\code{lam1.optimal}}{: optimal value for lambda_1.}
#'   \item{\code{lam2.optimal}}{: optimal value for lambda_2.}
#' }
#'
#'
#' @examples
#' p = 10
#' n = 50
#' sigma = matrix(1,nrow = p,ncol = p)*0.5 + diag(1,p,p)*0.5
#' data <- round(mvrnorm(n, mu = rep(0,p), sigma),2)         # generate data
#'
#' res <- cv.gconcord(data = data, lam1.vec = seq(0,1,0.02), # user-defined choices for lambda_1
#'                    lam2.vec = seq(0, 1, 0.02),            # user-defined choices for lambda_2
#'                    K = 5,                                 # 5-fold cross validation
#'                    method = "ista" )
#' res$lam1.optimal           # optimal lambda_1
#' res$lam2.optimal           # optimal lambda_2
#'
#' @useDynLib
#' @export
cv.gconcord <- function(data, rand, FUN,
                        lam1.vec, lam2.vec,
                        K = 3, method = "coordinatewise", tol = 1e-5, maxit = 100, steptype = 0)
{
  if(!require("plyr")){ install.packages("plyr")}
  if(missing(rand)){ rand <- sample( cut(seq.int(nrow(data)), breaks = K, labels = FALSE) ) }
  if(missing(FUN)){ FUN = pred.risk }
  if(missing(lam1.vec)){ lam1.vec = 10^seq(-1.2,0,0.05) - 10^(-1.2) }
  if(missing(lam2.vec)){ lam2.vec = (10^seq(-1.2,0,0.05) - 10^(-1.2))*10 }

  if(is.matrix(lam1.vec)){ A = 1 }else{ A = length(lam1.vec) }

  res <- matrix(0, nrow = A, ncol = length(lam2.vec))

  for(i in 1:A){
    lam1 = lam1.vec[i]
    for(j in 1:length(lam2.vec)){
      lam2 = lam2.vec[j]
      cost <- plyr::ldply(1:K, cost.func, data, rand, FUN, lam1, lam2, method, tol, maxit, steptype)
      res[i,j] <- mean(cost[,1])
    }
  }
  colnames(res) <- round(lam2.vec, 4)
  if(is.matrix(lam1.vec)){ rownames(res) <- -1 }else{ rownames(res) <- round(lam1.vec, 4) }

  res.quantile = quantile.map(res)

  idx = which(res == min(res), arr.ind = TRUE)

  if(is.matrix(lam1.vec)){lam1.opt = lam1.vec}else{ lam1.opt = lam1.vec[idx[1]]}
  lam2.opt = lam2.vec[idx[2]]

  return(list(val.error = structure(res, rows = "lambda1", cols = "lambda2"),
              val.error.quantile = structure(res.quantile, rows = "lambda1", cols = "lambda2"),
              lam1.optimal = lam1.opt,
              lam2.optimal = lam2.opt,
              lam1.cand = lam1.vec,
              lam2.cand = lam2.vec))
}

#'
#' Level Plot Function of Validation Error Matrix from Cross Validation
#'
#' This function is used to visualized the validation error matrix obtained from cross validation.
#'
#' @param mat the validation error matrix to be visualized.
#' @param col.region color vector to be used, see \code{\link[lattice]{levelplot}}. The default is
#'        \code{heat.colors(100)}
#' @param contour a logical flag, indicating whether to draw contour lines, see
#'        \code{\link[lattice]{levelplot}}.
#' @param ... other arguments in \code{\link[lattice]{levelplot}}.
#'
#' @examples
#' # Generate random data
#' p = 30
#' n = 25
#' sigma = matrix(1,nrow = p,ncol = p)*0.5 + diag(1,p,p)*0.5
#' data <- round(mvrnorm(n, mu = rep(0,p), sigma),2)
#'
#' # Cross validation for both parameters
#' res <- cv.gconcord(data = data, lam1.vec = seq(0,1,0.02), # user-defined choices for lambda_1
#'                    lam2.vec = seq(0, 1, 0.02),            # user-defined choices for lambda_2
#'                    K = 5,                                 # 5-fold cross validation
#'                    method = "ista" )
#' cvplot(res$val.error.quantile)
#'
#' # Cross validation for parameter lambda 1.
#' res <- cv.gconcord(data = data, lam1.vec = seq(0,1,0.02), # user-defined choices for lambda_1
#'                    lam2.vec = 0.05)                       # fixed lambda_2
#' cvplot(res$val.error.quantile)
#'
#' # Cross validation for parameter lambda 2.
#' res <- cv.gconcord(data = data, lam1.vec = 0.05,          # fixed lambda_1
#'                    lam2.vec = seq(0, 8, 0.4))             # user-defined choices for lambda_2                       # 3-fold cross validation
#' cvplot(res$val.error.quantile)
#'
#' @export
cvplot <- function(mat,
                   col.regions = heat.colors(100),
                   scales=list(x=list(rot=90)),
                   contour = TRUE, xlab, ylab, main, type, ...
                   ){
  if(missing(xlab)){xlab = expression(lambda[1])}
  if(missing(ylab)){ylab = expression(lambda[2])}
  if(missing(type)){type = "b"}
  if(dim(mat)[1] == 1){

    if(missing(main)){main = "Curve plot of cross validation result"}
    plot(1:length(mat[1,]), mat[1,], xaxt = 'n', type = type, xlab = ylab, ylab = "Loss value", main = main, ...)
    axis(side = 1, at = 1:length(mat[1,]), labels = colnames(mat))
    abline(v = which.min(mat), lty = 2, lwd = 2, col = "deepskyblue4")

  }else if (dim(mat)[2] == 1){

    if(missing(main)){main = "Curve plot of cross validation result"}
    plot(1:length(mat[,1]), mat[,1], xaxt = 'n', type = type, xlab = xlab, ylab = "Loss value", main = main, ...)
    axis(side = 1, at = 1:length(mat[,1]), labels = rownames(mat))
    abline(v = which.min(mat), lty = 2, lwd = 2, col = "deepskyblue4")

  }else{

    if(missing(main)){main = "Level plot of cross validation result"}
    lattice::levelplot(mat, col.regions = col.regions, scales = scales, contour = contour, main = main,
                       xlab = xlab, ylab = ylab, ...)

  }

}

#' Predictive Risk Loss Function in Cross Validation
#'
#' Calculate the predictive risk for a given estimator \emph{omega}.
#'
#' @param omega the estimated sparse inverse covariance matrix.
#' @param data data used in the loss function.
#'
#' @examples
#' p = 10
#' n = 50
#' sigma = matrix(1,nrow = p,ncol = p)*0.5 + diag(1,p,p)*0.5
#' data <- round(mvrnorm(n, mu = rep(0,p), sigma),2)         # generate data
#' omega <- gconcord(data, lambda1 = 0.2, lambda2 = 0.1, method = "ista")$omega
#' pred.risk(omega, data)
#'
#' @export
pred.risk <- function(omega, data){
  arg <- data %*% omega %*% solve( diag(diag(omega)) )
  return(norm(arg, "F") / nrow(data))
}

#'
#' Visualization of graphical model.
#'
#' Visualize the graphical model given an estimated inverse covariance matrix.
#'
#' @param met the estimated inverse covariance matrix.
#' @param varnames optional, a vector containing the labels of variables. If missing, variables will be
#' labeled by 1, 2, 3, ...
#' @param title optional, title of the plot.
#' @param seed optional, set the seed making result reproducible.
#' @param model optional, a placement method from those provided in the \code{link[GGally]{ggnet2}}
#' package: see \code{\link[sna]{gplot.layout}} for details. The default is "circle".
#' @param label optional, a logical value indicating if labels are shown in the graph. The default is \code{TRUE}.
#' @param edge.width optional, a positive number controling the width of edges. The default is 1.
#' @param color optional, color of the nodes in the graphs. See the argument \code{node.color} in \code{[GGally]{ggnet}}. The default is "lightpink".
#' @param edge.color optional, color of edges in the graphs. See the argument \code{node.color} in \code{[GGally]{ggnet}}. The default is "lightpink".
#' @param edge.size optional, size of the edges in the graphs. See the argument \code{size} in \code{[GGally]{ggnet}}. The default is "weights".
#' @param ... optional, other arguments. See \code{[GGally]{ggnet}}.
#'
#' @examples
#' p = 10
#' n = 50
#' sigma = matrix(1,nrow = p,ncol = p)*0.5 + diag(1,p,p)*0.5
#' data <- round(mvrnorm(n, mu = rep(0,p), sigma),2)         # generate data
#' train <- sample(1:nrow(data), 40, replace = FALSE)        # select training data
#' omega <- gconcord(data[train,], lambda1 = 0.2, lambda2 = 0.1, method = "ista")$omega
#' graphplot(omega)
#'
#' @export
graphplot <- function(met, varnames, main, seed, mode = "circle", label = TRUE,
                        edge.width = 1, color = "lightpink",
                        edge.color = "gray", edge.size = "weights", ...){
  # assign labels
  if(!require("GGally")){install.packages("GGally")}
  if(!require("network")){install.packages("network")}
  if(!require("sna")){ install.packages("sna")}
  if(!require("ggplot2")){ install.packages("ggplot2")}
  if(missing(varnames)){
    if(!is.null(colnames(met))){
      varnames <- colnames(met)
    }else if(!is.null(rownames(met))){
      varnames <- rownames(met)
    }else{
      varnames <- seq(1, ncol(met), 1)
    }
  }
  if(missing(main)){ main = "Sparse graph"}
  # visualization
  colnames(met) <- rownames(met) <- varnames
  A = abs(met)
  diag(A) <- rep(0, nrow(A))
  if(max(A) != 0){ c = 5 / max(A) * edge.width }else{ c = edge.width }
  if(!missing(seed)){ set.seed(seed) }
  net = network::network(abs(met)*c, directed = FALSE, matrix.type = "adjacency",
                         ignore.eval = FALSE, names.eval = "weights")
  if(!missing(seed)){ set.seed(seed) }
  library(sna)
  library(ggplot2)
  GGally::ggnet2(net, mode = mode, label = label, color = color,
         edge.color = "gray",edge.size = "weights",...) + ggtitle( main )
}

#'
#' Get daily returns of DJIA component stocks for a given time horizon.
#'
#' Return a data frame containing daily returns of DJIA component stocks for a given
#' time horizon.
#'
#' @param start optional, a string of starting date with format yyyy-mm-dd. The default
#' is "1990-01-03".
#' @param end optional, a string of ending date with format yyyy-mm-dd. The default is
#' "2018-02-28". Ending date should be no earlier than starting date.
#' @param type optional, a string indicating the type of DJIA data to be extracted. Defalut
#' value is \code{return}, the daily return of 30 component stocks of DJIA. Available choices are:
#' \itemize{
#'   \item{\code{return}}{: daily arithmatic return of all DJIA component stocks. Maximum time
#'   horizon is from 1990-01-03 to 2018-02-28.}
#'   \item{\code{price}}{: daily closing prices of all DJIA component stocks. Maximum time
#'   horizon is from 1990-01-02 to 2018-02-28.}
#'   \item{\code{index}}{: daily closing price of DJIA Index. Maximum time horizon is from
#'   1990-01-02 to 2018-03-02.}
#' }
#' @param na.rm optional, a boolean indicating if the column containing a missing value
#' is removed. The default is \code{TRUE}.
#'
#' @examples
#' # Get return data
#' data = get.data(start = "2017-01-01", end = "2017-12-31", type = "return", na.rm = TRUE)
#'
#' # Get price data
#' data = get.data(start = "2005-04-15", end = "2005-04-30", type = "price", na.rm = FALSE)
#'
#' # Get DJIA index data
#' data = get.data(start = "1998-10-27", end = "1999-01-01", type = "index")
#'
#' @export
get.data <- function(start = "1990-01-03", end = "2018-02-28", type = "return", na.rm = TRUE){
  if(!require("lubridate")){ install.packages("lubridate")}
  if(type == "price"){
    data = gconcord::DJPrice
  }else if(type == "index"){
    data = gconcord::DJIndex
  }else{
    data = gconcord::return
  }
  if(type != "index"){alldates = lubridate::date(rownames(data)) }else{alldates = lubridate::date(names(data))}
  sdate = as.Date( start )
  edate = as.Date( end )
  start = min(which(as.Date(alldates) - sdate >= 0))
  end = max(which(as.Date(alldates) - edate <= 0))
  if(type == "index"){ return(data[start:end]) }
  if( na.rm ){
    del <- which(is.na(colMeans(data[start:end,])))
    if(length(del) != 0){
      return( data[start:end,-del] )
    }
  }
  return( data[start:end,])
}

#'
#' Get sparsity level of a matrix
#'
#' Get the sparsity level of a matrix, measured by either a ratio of number of zero
#' elements over total number of elements, or the count of zero elements.
#'
#' @param mat a matrix input
#' @param ratio optional, a boolean value indicating if function returns a ratio of
#' number of zero elements over total number of elements, or the count of zero elements.
#' The default is \code{TRUE}.
#' @param thres optional, a threshold of determining a zero element. The default is 1e-5.
#'
#' @examples
#' mat <- diag(5)
#' zero.count(mat)
#' zero.count(mat, ratio = FALSE)
#'
#' @export
zero.count <- function( mat, ratio = TRUE, thres = 1e-5 ){

  count = 0
  n = nrow(mat)
  p = ncol(mat)
  for(i in 1:n){
    for(j in 1:p){
      if( abs(mat[i,j]) < thres ){
        count = count + 1
      }
    }
  }

  if(ratio){
    return(count / (n*p))
  }else{
    return(count)
  }
}
