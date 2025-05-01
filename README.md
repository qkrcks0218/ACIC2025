ACIC2025 Short Course: Proximal Causal Inference
================
Chan Park
2025-05





This document is part of the ACIC2025 short course and demonstrates the
identification and estimation of the average treatment effect using
proximal causal inference.

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
```

### Proximal 2-stage regression procedure

We consider the following linear models:

$$
\begin{aligned}
E(Y|D,Z,U) & = &  \beta_0 & +  \beta_D D & +  \beta_U U
\\
E(W|D,Z,U) & = &  \gamma_0 &  &  +  \gamma_U U
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
W.fit <- lm(as.formula(paste("cbind(ph1,hema1)~RHC+",paste0(Z,"+",X,collapse = "+"))),
            data=RHC_data_reform)
RHC_data_reform_WfitAdd <- RHC_data_reform
RHC_data_reform_WfitAdd$W.fit <- predict(W.fit) 

## 2st stage
Y.fit <- lm(as.formula(paste("Y~RHC+W.fit+",X)), data=RHC_data_reform_WfitAdd)

summary(Y.fit)$coefficients[1:5,]
```

    ##                 Estimate  Std. Error   t value     Pr(>|t|)
    ## (Intercept) 156.24019359 57.44375771  2.719881 6.550466e-03
    ## RHC          -1.99314108  0.37796646 -5.273328 1.389440e-07
    ## W.fitph1    -18.41544172  7.52248014 -2.448055 1.439313e-02
    ## W.fithema1   -1.15908980  0.59952519 -1.933346 5.324334e-02
    ## age           0.05314342  0.03452365  1.539334 1.237788e-01

Unfortunately, the reported standard error (SE) is typically smaller
than the true SE, as it does not account for the variability from the
first-stage regression.

In fact, a one-line implementation is possible using the `ivreg::ivreg`
function, which also provides a valid SE.

``` r
IVReg <- ivreg::ivreg(as.formula(paste("Y~RHC+ph1+hema1+",X,"|",
                                       paste0(Z,"+RHC+",X,collapse = "+"))),
                      data=RHC_data_reform)
summary(IVReg)$coefficients[1:5,]
```

    ##                 Estimate  Std. Error   t value     Pr(>|t|)
    ## (Intercept) 156.24019359 76.69182462  2.037247 4.167169e-02
    ## RHC          -1.99314108  0.50461422 -3.949831 7.915611e-05
    ## ph1         -18.41544172 10.04308824 -1.833643 6.675941e-02
    ## hema1        -1.15908980  0.80041214 -1.448116 1.476399e-01
    ## age           0.05314342  0.04609172  1.152993 2.489621e-01

The effect estimate of -1.99 is statistically significant at the 5%
level, indicating that RHC reduces survival time. In comparison, the OLS
estimate is smaller than that obtained from the two-stage proximal
regression procedure.

``` r
OLS <- lm(as.formula(paste("Y~RHC+",W,"+",Z,"+",X)), data=RHC_data_reform)
summary(OLS)$coefficients[1:5,]
```

    ##                 Estimate   Std. Error    t value     Pr(>|t|)
    ## (Intercept) -9.873021982 11.659335728 -0.8467911 3.971474e-01
    ## RHC         -1.326837888  0.283105483 -4.6867262 2.841134e-06
    ## ph1          3.323119542  1.446673158  2.2970769 2.165071e-02
    ## hema1       -0.030746971  0.017362840 -1.7708492 7.663959e-02
    ## pafi1        0.002696037  0.001231942  2.1884448 2.867794e-02

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

We will consider a linear $h$ and a linear $g_h$:

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

Although these specifications can be changed, we keep them simple for
demonstration purposes.

The ATE is identified by $\beta = E [ h(D=1,W,X) -h(D=0,W,X) ]$.

### Estimation of the ATE using the outcome confounding bridge function

We consider a moment function $\Psi(O,\beta,\theta)$:

$$
\Psi(O,\beta,\theta) = \left[ \matrix{ h(1,W,X; \theta)-h(0,W,X; \theta) - \beta  
\\ 
g_h(D,Z,X) ( Y - h(D,W,X; \theta) ) } \right]
$$

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
```

Consider the solution to the moment equation

$$
(\widehat{\beta},\widehat{\theta})
\quad \leftarrow \quad 
\frac{1}{N} \sum_{i=1}^{N} \Psi(O_i , \beta,\theta) = 0
$$

where $\widehat{\beta}$ is an ATE estimate. The moment equation can be
solved by the off-the-shelf optimization function `optim`.

``` r
sum.extended.moment <-            ## average of Psi over the observations
  function(data,theta.extended){
    sum(apply(extended.moment(data,theta.extended),2,mean)^2)
  }

Moment.Equation <-                ## Find the solutions of the moment equation
  optim(par=as.vector(c(IVReg$coefficients[2],IVReg$coefficients)),
        fn=function(theta){sum.extended.moment(data=RHC_data_reform,theta)})

Moment.Equation$par[1]           ## ATE 
```

    ## [1] -1.993141

``` r
cbind(Moment.Equation$par[1+1:5],
      IVReg$coefficients[1:5])   ## (ATE,coef,RHC,ph1,hema1,age)
```

    ##                     [,1]         [,2]
    ## (Intercept) 156.24019359 156.24019359
    ## RHC          -1.99314108  -1.99314108
    ## ph1         -18.41544172 -18.41544172
    ## hema1        -1.15908980  -1.15908980
    ## age           0.05314342   0.05314342

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

RESULT <- cbind(c(Moment.Equation$par[1], sqrt(Avar[1,1]), 
                  Moment.Equation$par[1]-1.96*sqrt(Avar[1,1]),
                  Moment.Equation$par[1]+1.96*sqrt(Avar[1,1])),
                c(summary(IVReg)$coefficients[2,1],summary(IVReg)$coefficients[2,2],
                  summary(IVReg)$coefficients[2,1]-1.96*summary(IVReg)$coefficients[2,2],
                  summary(IVReg)$coefficients[2,1]+1.96*summary(IVReg)$coefficients[2,2]))

colnames(RESULT) <- c("Bridge Ft","2SLS")
rownames(RESULT)  <- c("Est","SE","95% CI LB","95% CI UB")
RESULT
```

    ##            Bridge Ft       2SLS
    ## Est       -1.9931411 -1.9931411
    ## SE         0.5142077  0.5046142
    ## 95% CI LB -3.0009881 -2.9821850
    ## 95% CI UB -0.9852940 -1.0040972

The result is similar to that using `ivreg`.

## Reference

Tchetgen Tchetgen et al. (2024) An Introduction to Proximal Causal
Inference. *Statistical Science*.

Liu, Park, Li, Tchetgen Tchetgen (2024). Regression-Based Proximal
Causal Inference. *Am J Epi*.

Stefanski, Boos (2002). The Calculus of M-Estimation. *The American
Statistician*.
