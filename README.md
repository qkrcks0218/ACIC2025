ACIC2025 Short Course: Proximal Causal Inference
================





This document is part of the ACIC2025 short course and demonstrates the
proximal causal inference framework, focusing on identification and
estimation of the average treatment effect.

## Proximal Identification and Estimation of the Average Treatment Effect (ATE)

### Dataset

We use the Study to Understand Prognoses Preferences Outcomes and Risks
of Treatment (SUPPORT) dataset. The variables included in the analysis
are listed below.

- $Y$:    Survival date (clipped at 30 days)
- $D$:    Right heart catheterization (RHC)
- $Z$:    `pafi1`, `paco21`
- $W$:   `ph1`, `hema1`
- $X$:    Other 68 variables (age, sex, weight, medical history, etc.)

The outcome variable $Y$ can be obtained from `rhc` dataset in `Hmisc`
library. The other variables are available from `RHC` dataset in
`ATbounds` library. See [this
website](https://hbiostat.org/data/repo/rhc.html) for details on the
dataset.

``` r
library(Hmisc)
```

    ## 
    ## Attaching package: 'Hmisc'

    ## The following objects are masked from 'package:base':
    ## 
    ##     format.pval, units

``` r
getHdata(rhc)     
library(ATbounds) 

RHC$Y <- apply(cbind(rhc$lstctdte - rhc$sadmdte,30),1,min)  ## survival date clipped at 30 days
Z_ls <- c("pafi1", "paco21")
W_ls <- c("ph1", "hema1")
D_ls <- "RHC"
X_ls <- colnames(RHC)[3:74]
X_ls <- X_ls[! X_ls %in% c(D_ls,Z_ls,W_ls)]
nX <- length(X_ls)

RHC_data_reform <- RHC[,c("Y",D_ls,W_ls,Z_ls,X_ls)]  ## Only use relevant variables
round(RHC_data_reform[1:10,1:10],3)
```

    ##     Y RHC   ph1  hema1   pafi1 paco21    age    edu cardiohx chfhx
    ## 1  30   0 7.359 58.000  68.000     40 70.251 12.000        0     0
    ## 2  30   1 7.329 32.500 218.312     34 78.179 12.000        1     1
    ## 3  30   1 7.359 21.098 275.500     16 46.092 14.070        0     0
    ## 4  30   0 7.460 26.297 156.656     30 75.332  9.000        0     0
    ## 5   1   1 7.229 24.000 478.000     17 67.910  9.945        0     0
    ## 6  30   0 7.300 30.500 184.188     68 86.078  8.000        0     1
    ## 7  30   0 7.380 29.000 148.750     45 54.968 14.000        0     0
    ## 8  30   0 7.560 33.594 240.000     26 43.639 12.000        0     0
    ## 9  30   0 7.400 18.898 333.312     40 18.042 12.822        0     0
    ## 10 30   1 7.350 32.695  68.000     30 48.424 11.041        0     0

### Proximal 2-stage regression procedure

We consider the following linear models:

$$
\begin{aligned}
E(Y|D,Z,U) & = &  \beta_0 & +  \beta_D D & +  \beta_X X & + \beta_U U
\\
E(W|D,Z,U) & = &  \gamma_0 &  & +  \gamma_X X &  + \gamma_U U
\end{aligned}
$$

where $\beta_D$ encodes the average treatment effect of $D$ on $Y$.

[Tchetgen Tchetgen et al. (2024, Stat.
Sci.)](https://projecteuclid.org/journals/statistical-science/volume-39/issue-3/An-Introduction-to-Proximal-Causal-Inference/10.1214/23-STS911.short)
and [Liu et al. (2024,
AJE)](https://academic.oup.com/aje/advance-article/doi/10.1093/aje/kwae370/7775568)
showed that a consistent estimator of $\beta_D$ can be obtained by the
following 2-stage regression procedure:

$$
\begin{aligned} 1. \quad & \widehat{W} & \leftarrow &  \quad \texttt{lm}(W \sim D +Z+X) 
\\ 
2. \quad &  \widehat{\beta}_{D} & \leftarrow & \quad \texttt{coef}( \texttt{lm}(Y \sim D +\widehat{W}+X) ) \end{aligned}
$$

This can be easily implemented using the `lm` function.

``` r
## concatenating variables to make formulas
Z <- paste0(Z_ls,collapse ="+")
W <- paste0(W_ls,collapse ="+")
X <- paste0(X_ls,collapse ="+")

## 1st stage
# lm(W~D+Z+X)
P2S.W.fit <- lm(as.formula(paste("cbind(ph1,hema1)~RHC+",Z,"+",X)),
            data=RHC_data_reform)
RHC_data_reform_WfitAdd <- RHC_data_reform
RHC_data_reform_WfitAdd$W.fit <- predict(P2S.W.fit) 

## 2nd stage
# lm(Y~D+W.hat+X)
P2S.Y.fit <- lm(as.formula(paste("Y~RHC+W.fit+",X)), data=RHC_data_reform_WfitAdd)

round(summary(P2S.Y.fit)$coefficients[1:5,],3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)  156.240     57.444   2.720    0.007
    ## RHC           -1.993      0.378  -5.273    0.000
    ## W.fitph1     -18.415      7.522  -2.448    0.014
    ## W.fithema1    -1.159      0.600  -1.933    0.053
    ## age            0.053      0.035   1.539    0.124

Unfortunately, the reported standard error (SE) is typically smaller
than the true SE, as it does not account for the variability from the
first-stage regression.

In fact, a one-line implementation is possible using the `ivreg::ivreg`
function, which also provides a valid SE.

``` r
## ivreg(Y~D+W+X|D+Z+X,data)
IVReg <- ivreg::ivreg(as.formula(paste("Y~RHC+ph1+hema1+",X,"|","RHC+",Z,"+",X)),
                      data=RHC_data_reform)
round(summary(IVReg)$coefficients[1:5,],3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)  156.240     76.692   2.037    0.042
    ## RHC           -1.993      0.505  -3.950    0.000
    ## ph1          -18.415     10.043  -1.834    0.067
    ## hema1         -1.159      0.800  -1.448    0.148
    ## age            0.053      0.046   1.153    0.249

Another one-line implementation is available using the `gmm::gmm`
function:

``` r
## gmm(Y~D+W+X,~D+Z+X,data)
GMM <- gmm::gmm(as.formula(paste("Y~RHC+ph1+hema1+",X)),
                as.formula(paste("~RHC+",Z,"+",X)),
                data=RHC_data_reform)
round(summary(GMM)$coefficients[1:5,],3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)  156.240     80.354   1.944    0.052
    ## RHC           -1.993      0.503  -3.961    0.000
    ## ph1          -18.415     10.535  -1.748    0.080
    ## hema1         -1.159      0.835  -1.388    0.165
    ## age            0.053      0.047   1.125    0.261

The effect estimate of -1.99 is statistically significant at the 5%
level, indicating that RHC reduces survival time.

In comparison, the OLS estimate is smaller than that obtained from the
two-stage proximal regression procedure.

``` r
## lm(Y~D+W+Z+X)
OLS <- lm(as.formula(paste("Y~RHC+",W,"+",Z,"+",X)), data=RHC_data_reform)
round(summary(OLS)$coefficients[1:5,],3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)   -9.873     11.659  -0.847    0.397
    ## RHC           -1.327      0.283  -4.687    0.000
    ## ph1            3.323      1.447   2.297    0.022
    ## hema1         -0.031      0.017  -1.771    0.077
    ## pafi1          0.003      0.001   2.188    0.029

### Outcome confounding bridge function

Recall that the outcome confounding bridge function $h$ satisfies

$$
\begin{aligned}
&
E[h(D,W,X)|D,Z,X] = E[Y|D,Z,X]
\\
\Rightarrow \quad &
E[ 
g_h(D,Z,X) ( Y - h(D,W,X) )
] = 0
\end{aligned}
$$

We will consider a linear $h$ and a linear $g_h$. Although these
specifications can be changed, we keep them simple for demonstration
purposes.

$$
\begin{aligned}
& h(D,W,X) = \theta_0 + \theta_D D + \theta_W^\intercal W + \theta_X^\intercal X
\\
& g_h(D,Z,X) = \left[ \matrix{1
\\ 
D
\\
Z
\\
X} \right]
\in \mathbb{R}^{1+1+2+68}
\end{aligned}
$$

``` r
h <- function(data,theta.h){
  ## data = [Y,D,W,Z,X]
  ## h1(d,w,x) = b0 + b1*d + b2*w + b3*x
  return( as.vector(
    as.matrix(cbind(1,data[,c(2, 3:4, 6+1:nX)]))%*%theta.h # [1,D,W,X]*theta
  ) )
}

moment.h <- function(data,theta.h){
  ## data = [Y,D,W,Z,X]
  ## E[ g_h(D,Z,X) * {Y-h(D,W,X)} ] = 0
  return( as.matrix(
    cbind(1,data[,c(2,5:6,6+1:nX)])*                # g_h(D,Z,X)
      as.vector((data[,1]-h(data,theta.h)))         # Y-h(D,W,X)
  ) )
}
```

The ATE is identified by

$$
\begin{aligned}
\beta & = E [ h(D=1,W,X) -h(D=0,W,X) ] 
\\
& = E [ ( \theta_0 + \theta_D  + \theta_W^\intercal W + \theta_X^\intercal X ) - ( \theta_0 + \theta_W^\intercal W + \theta_X^\intercal X )  ] 
\\
& = \theta_D
\end{aligned}
$$

Note that, under the linear $h$, we have $\beta = \theta_D$

### Estimation of the ATE using the outcome confounding bridge function

We consider a moment function $\Psi(O,\beta,\theta)$:

$$
\Psi(O,\beta,\theta) = \left[ \matrix{ h(1,W,X; \theta)-h(0,W,X; \theta) - \beta  
\\ 
g_h(D,Z,X) ( Y - h(D,W,X; \theta) ) } \right]
$$

Note that the solution to the population moment equation
$E[\Psi(O,\beta,\theta)]=0$ will recover the ATE $(=\beta)$ and the
outcome confounding bridge function parameters $(=\theta)$. Therefore,
estimators of $\beta$ and $\theta$ can be constructed as the solutions
to the moment equation based on the observed data:

$$
(\widehat{\beta},\widehat{\theta})
\quad \leftarrow \quad 
\frac{1}{N} \sum_{i=1}^{N} \Psi(O_i , \beta,\theta) = 0
$$

The moment equation can be solved by the off-the-shelf optimization
function `optim`.

``` r
extended.moment <-                ## Psi function
  function(data,theta.extended){    
    ## data = [Y,D,W,Z,X]
    theta.ate <- theta.extended[1]
    theta.h <- theta.extended[1+1:(4+nX)]
    data1 <- data0 <- data
    data1[,2] <- 1
    data0[,2] <- 0
    cbind( h(data1,theta.h)-h(data0,theta.h)-theta.ate,
           moment.h(data,theta.h) )
  }

sum.extended.moment <-            ## (average of Psi over the observations)^2
  function(data,theta.extended){
    sum(apply(extended.moment(data,theta.extended),2,mean)^2)
  }

Moment.Equation <-                ## Find the solutions of the moment equation
  optim(par=as.vector(c(IVReg$coefficients[2],IVReg$coefficients)),
        fn=function(theta){sum.extended.moment(data=RHC_data_reform,theta)})

round(Moment.Equation$par[1],3)           ## β
```

    ## [1] -1.993

``` r
Theta <- cbind(P2S.Y.fit$coefficients,
               IVReg$coefficients[1:5],
               GMM$coefficients[1:5],
               Moment.Equation$par[1+1:5])
```

    ## Warning in cbind(P2S.Y.fit$coefficients, IVReg$coefficients[1:5],
    ## GMM$coefficients[1:5], : number of rows of result is not a multiple of vector
    ## length (arg 2)

``` r
colnames(Theta) <- c("P2S","ivreg","gmm","Bridge Ft")
round(Theta[1:5,],3)   ## θ
```

    ##                 P2S   ivreg     gmm Bridge Ft
    ## (Intercept) 156.240 156.240 156.240   156.240
    ## RHC          -1.993  -1.993  -1.993    -1.993
    ## W.fitph1    -18.415 -18.415 -18.415   -18.415
    ## W.fithema1   -1.159  -1.159  -1.159    -1.159
    ## age           0.053   0.053   0.053     0.053

Since $h(D=1,W,X)-h(D=0,W,X) = \theta_D$ under the linear $h$, we find
the coefficients of RHC is equal to the ATE estimate.

A corresponding variance estimator can also be constructed; we refer the
readers to [Stefanski and Boos (2002, The Amer.
Stat.)](https://www.jstor.org/stable/3087324) for details.

``` r
AVAR <- function(data,theta.extended,extended.moment){
  
  Gradient <- function(data,theta.E){
    d <- length(theta.E)
    m <- ncol(extended.moment(data,theta.E))
    G <- matrix(0,d,m)
    for(ii in 1:d){
      pert <- rep(0,d)
      pert[ii] <- pert[ii]+10^(-8)
      G[ii,] <- apply((extended.moment(data,theta.E+pert) - 
                         extended.moment(data,theta.E-pert))/(2*10^(-8)),2,mean)
    }
    return(G) # d*m 
  }
  
  Grad <- Gradient(data,theta.extended)
  Res.Sq <- t(extended.moment(data, theta.extended))%*%
    extended.moment(data,theta.extended)/nrow(data)
  SolveG <- MASS::ginv(Grad, tol=10^(-12))
  Avar <- t(SolveG)%*%Res.Sq%*%(SolveG)/nrow(data)
  return(Avar)
}

Avar <- AVAR(RHC_data_reform, 
             Moment.Equation$par,
             extended.moment)

RESULT <- cbind(
  
  c(summary(OLS)$coefficients[2,1],
    summary(OLS)$coefficients[2,2],
    summary(OLS)$coefficients[2,1]-1.96*summary(OLS)$coefficients[2,2],
    summary(OLS)$coefficients[2,1]+1.96*summary(OLS)$coefficients[2,2]),
  
  c(summary(P2S.Y.fit)$coefficients[2,1],
    summary(P2S.Y.fit)$coefficients[2,2], ## Invalid
    summary(P2S.Y.fit)$coefficients[2,1]-1.96*summary(P2S.Y.fit)$coefficients[2,2],  ## Invalid
    summary(P2S.Y.fit)$coefficients[2,1]+1.96*summary(P2S.Y.fit)$coefficients[2,2]), ## Invalid
  
  c(summary(IVReg)$coefficients[2,1],
    summary(IVReg)$coefficients[2,2],
    summary(IVReg)$coefficients[2,1]-1.96*summary(IVReg)$coefficients[2,2],
    summary(IVReg)$coefficients[2,1]+1.96*summary(IVReg)$coefficients[2,2]),
  
  c(summary(GMM)$coefficients[2,1],
    summary(GMM)$coefficients[2,2],
    summary(GMM)$coefficients[2,1]-1.96*summary(GMM)$coefficients[2,2],
    summary(GMM)$coefficients[2,1]+1.96*summary(GMM)$coefficients[2,2]),
  
  c(Moment.Equation$par[1], 
    sqrt(Avar[1,1]), 
    Moment.Equation$par[1]-1.96*sqrt(Avar[1,1]),
    Moment.Equation$par[1]+1.96*sqrt(Avar[1,1])))

colnames(RESULT) <- c("OLS","P2S","ivreg","gmm","Bridge Ft")
rownames(RESULT)  <- c("Est","SE","95% CI LB","95% CI UB")
round(RESULT,3)
```

    ##              OLS    P2S  ivreg    gmm Bridge Ft
    ## Est       -1.327 -1.993 -1.993 -1.993    -1.993
    ## SE         0.283  0.378  0.505  0.503     0.514
    ## 95% CI LB -1.882 -2.734 -2.982 -2.979    -3.001
    ## 95% CI UB -0.772 -1.252 -1.004 -1.007    -0.985

The result from the bridge function approach is similar to that using
`ivreg` or `gmm` because $h$ is linear in $D$. Note that the SE of the
proximal 2-stage regression procedure is invalid.

## Reference

Tchetgen Tchetgen et al. (2024) An Introduction to Proximal Causal
Inference. *Statistical Science*.

Liu, Park, Li, Tchetgen Tchetgen (2024). Regression-Based Proximal
Causal Inference. *American Journal of Epidemiology*.

Stefanski, Boos (2002). The Calculus of M-Estimation. *The American
Statistician*.
