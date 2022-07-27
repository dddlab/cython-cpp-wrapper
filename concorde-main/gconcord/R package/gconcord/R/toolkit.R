
is.scalar <- function( x ){
  ## Return Boolean value if x is a scalar number or not.
  return( is.atomic(x) && length(x) == 1L )
}


check.penalty <- function( lambda1, lambda2, p ){

  ## Check penalty input

  try(if( !is.scalar(lambda2) ) stop("lambda2 is not scalar."))

  if(is.scalar(lambda1)){
    lam1 <- matrix(lambda1, p, p) - diag(lambda1, p)
  }else{
    lam1 <- lambda1 - diag(diag(lambda1))
  }

  return( list(lam1 = lam1, lam2 = lambda2) )

}


cost.func <- function(chunkid, data, rand, FUN, lam1, lam2, method, tol, maxit, steptype){

  train = (rand != chunkid)
  Dtr = data[train,]
  Dvl = data[!train,]
  omega <- do.call(gconcord, list(data = Dtr, lambda1 = lam1,
                                  lambda2 = lam2, method = method, tol = tol,
                                  maxit = maxit, steptype = steptype))
  return( do.call(FUN, list(omega = omega, data = Dvl)) )
}


quantile.map <- function(res){
  percentile <- ecdf( as.vector(res) - (1e-05) )
  res.quantile <- matrix(0, nrow = nrow(res), ncol = ncol(res))
  for(i in 1:nrow(res)){
    for(j in 1:ncol(res)){
      res.quantile[i,j] = percentile(res[i,j])
    }
  }
  rownames(res.quantile) = rownames(res)
  colnames(res.quantile) = colnames(res)
  return(res.quantile)
}



