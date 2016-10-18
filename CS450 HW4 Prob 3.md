# CS 450 Homework 4 Problem \# 1
## Mohammad Saad

### _Critical Points in 2 Dimensions_

Determine the critical points of each of the following functions and characterize each as a minimum, maximum, or saddle point. Also determine whether each function has a global minimum or maximum on $ℝ^2$.

1. $ f(x,y) = x^2 - 4xy + y^2$
Taking the gradient with respect to $x$ and $y$, we see that 
$\frac{df}{dx} = 2x - 4y$ 
$\frac{df}{dy} = 2y - 4x$ 
Setting them equal to 0 and solving the system, we see that the critical point resides at $(0,0)$. The critical point must is a saddle point - there are values greater and lower then the function as we vary $x$ and $y$.

2. $ f(x,y) = x^4 - 4xy + y^4$
Our gradient will be:
$$
\begin{bmatrix}
\frac{df}{dx}\\
\frac{df}{dy}\\
\end{bmatrix} = 
\begin{bmatrix}
4x^3 - 4y\\
4y^3 - 4x\\
\end{bmatrix} 
$$
Setting this system equivalent to 0, we see that we have three critical points: $(0,0), (1,1),(-1,-1)$. To check the critical points, we compute the Hessian:
$$
{H_f(x)} = \begin{bmatrix}
(4x^3 - 4y)^2 && (4x^3 - 4y)(4y^3 - 4x)\\
(4x^3 - 4y)(4y^3 - 4x)&& (4y^3 - 4x)\\
\end{bmatrix} 
$$
Doing this, we find that (0,0) is a saddle point, while (1,1) and (-1,1) are global minima.

3. $f(x,y) = 2x^3 - 3x^2 - 6xy(x - y - 1)$
Our gradient is
$$
\begin{bmatrix}
\frac{df}{dx}\\
\frac{df}{dy}\\
\end{bmatrix} = 
\begin{bmatrix}
6x^2 - 6x  - 12xy + 6y^2 +  6y\\
-6x^2 + 12xy + 6x\\
\end{bmatrix} 
$$
Solving for 0, we find that our critical points are (0,0), (0,1), (0, -1), (-1,-1). We can place them in a Hessian Matrix:
$$
{H_f(x)} = \begin{bmatrix}
12x - 12y - 6 && 12y - 12x + 6\\
12y - 12x - 6 && 12x\\
\end{bmatrix} 
$$
Then we can plug in our critical points and see that (0,0) is a saddle point,  (0,1) is a saddle point, (0, -1) is a saddle point, and (-1,-1) is a maximum, but not a global one. 

4. $f(x,y) = (x-y)^4 + x^2 - y^2 - 2x + 2y + 1$
Our gradient is
$$
\begin{bmatrix}
\frac{df}{dx}\\
\frac{df}{dy}\\
\end{bmatrix} = 
\begin{bmatrix}
2x  + 4(x-y)^3 - 2\\
2 - 4(x-y)^3 - 2y\\
\end{bmatrix} 
$$
Solving for 0, we find that our critical point are (1,1). We can place them in a Hessian Matrix:
$$
{H_f(x)} = \begin{bmatrix}
12(x-y)^2 + 2 && -12(x-y)^2\\
-12(x-y)^2 && 12(x-y)^2 - 2\\
\end{bmatrix} 
$$

We also find that this is a local minimum and a saddle point.


 