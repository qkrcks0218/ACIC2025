Proximal Causal Inference
================
Chan Park
2025-05





This demonstration will discuss inference of the average treatment
effect and the natural direct and indirect effects using proximal causal
inference.

## Inference of the Average Treatment Effect (ATE)

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
`ATbounds` library.

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
X_ls <- X_ls[1:3]
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

TT et al. (2024, Stat. Sci.) and Liu et al. (2024, AJE) showed that a
consistent estimator of $\beta_D$ can be obtained by the following
2-stage regression procedure:

$$
\begin{aligned}
1. \quad &
\widehat{W} & \quad \leftarrow &  \quad \texttt{lm}(W \sim D +Z+X)
\\
2. \quad &
\widehat{\beta}_{D} &
\quad \leftarrow & \quad \texttt{coef}( \texttt{lm}(Y \sim D +\widehat{W}+X) )
\end{aligned}
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

    ##                  Estimate  Std. Error   t value     Pr(>|t|)
    ## (Intercept) -2016.5613353 640.0329864 -3.150715 1.637106e-03
    ## RHC            18.9391389   6.1152582  3.097030 1.964153e-03
    ## W.fitph1      245.0493156  77.4317521  3.164714 1.560486e-03
    ## W.fithema1      7.7677238   2.2644965  3.430221 6.073471e-04
    ## age            -0.4026371   0.0991632 -4.060348 4.965303e-05

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

    ##                  Estimate   Std. Error    t value  Pr(>|t|)
    ## (Intercept) -2016.5613355 4304.0632272 -0.4685250 0.6394270
    ## RHC            18.9391389   41.1235959  0.4605419 0.6451448
    ## ph1           245.0493156  520.7093447  0.4706067 0.6379395
    ## hema1           7.7677238   15.2281776  0.5100889 0.6100089
    ## age            -0.4026371    0.6668479 -0.6037915 0.5460062

The effect estimate of -1.99 is statistically significant at the 5%
level, indicating that RHC reduces survival time. In comparison, the OLS
estimate is smaller than that obtained from the two-stage proximal
regression procedure.

``` r
OLS <- lm(as.formula(paste("Y~RHC+",W,"+",Z,"+",X)), data=RHC_data_reform)
summary(OLS)$coefficients[1:5,]
```

    ##                  Estimate   Std. Error   t value     Pr(>|t|)
    ## (Intercept) -50.942110272 10.550040460 -4.828618 1.410590e-06
    ## RHC          -0.900381381  0.288489946 -3.121015 1.811260e-03
    ## ph1           9.623570692  1.397238341  6.887566 6.283741e-12
    ## hema1         0.045819436  0.016701528  2.743428 6.099069e-03
    ## pafi1         0.004180785  0.001211658  3.450467 5.636574e-04

### Outcome confounding bridge function

Recall that the outcome confounding bridge function $h$ satisfies

$$
\begin{aligned}
&
E[h(D,W,X)|D,Z,X] = E[Y|D,Z,X]
\\
\Rightarrow \quad &
E[ 
g_h(D,Z,X) \{ Y - h(D,W,X) \}
] = 0
\end{aligned}
$$

We will consider a linear $h$ and a linear $g_h$:

$$
\begin{aligned}
& h(D,W,X) = \theta_0 + \theta_D D + \theta_W^\intercal W + \theta_X^\intercal X
\\
& g_h(D,Z,X) = 
[ 1, D, Z^\intercal, X^\intercal]^\intercal
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

We consider a moment function $\Psi(O,\beta,\theta)$:

$$
\Psi(O,\beta,\theta) = \left[ \matrix{ h(1,W,X)-h(0,W,X) - \beta  \\ g_h(D,Z,X) \{ Y - h(D,W,X; \theta) \}} \right]
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

Note that $\widehat{\beta}$ is an ATE estimate.

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

    ## [1] 18.93914

``` r
cbind(Moment.Equation$par[1+1:5],
      IVReg$coefficients[1:5])   ## (ATE,coef,RHC,ph1,hema1,age)
```

    ##                      [,1]          [,2]
    ## (Intercept) -2016.5613355 -2016.5613355
    ## RHC            18.9391389    18.9391389
    ## ph1           245.0493156   245.0493156
    ## hema1           7.7677238     7.7677238
    ## age            -0.4026371    -0.4026371

Since $h(D=1,W,X)-h(D=0,W,X) = \theta_D$ under the linear $h$, we find
the coefficients of RHC is equal to the ATE estimate.

A corresponding variance estimator can also be constructed. The result
is similar to that using `ivreg`.

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

    ##           Bridge Ft      2SLS
    ## Est        18.93914  18.93914
    ## SE         44.52830  41.12360
    ## 95% CI LB -68.33632 -61.66311
    ## 95% CI UB 106.21460  99.54139
