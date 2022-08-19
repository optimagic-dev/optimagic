# Richardson Extrapolation

In this section we introduce the mathematical machinery of *Richardson's method*.

## Motivation

Say you want to compute the value of some function $g: \mathbb{R}_+ \to
\mathbb{R}^{m\times n}, h \mapsto g(h)$ as $h \to 0$; however,
$\lim_{h\to\infty} g(h)\neq g(0)$. We can approximate the limit by evaluating the
function at values close to zero on a computer.  The error of our approximation
naturally depends on $g$. In certain cases it is possible to express this error
in a specific way, in which case we can improve upon the order of our error using
Richardson's method.

### Example

Lets start with an easy case where $f: \mathbb{R} \to \mathbb{R}$ is the function
of interest. Using central differences we can approximate $f'$ at some point
$x \in \mathbb{R}$ by $g(h) := \frac{f(x+h) - f(x-h)}{2h}$. Note that
$g(h) \to f'(x)$ as $h \to 0$ if $f$ is differentiable at $x$;
however, $g(0)$ is not defined and hence in particular unequal to $f'(x)$.
To quantify the error of using $g(h)$ instead of $f'(x)$ we can rely on
Taylor's Theorem (assuming that $f$ has a Taylor representation):

$$
f(x+h) &= f(x) + f'(x)h + f''(x)\frac{h^2}{2} + f'''(x)\frac{h^3}{6} +
\dots\\ f(x-h) &= f(x) - f'(x)h + f''(x)\frac{h^2}{2} -
f'''(x)\frac{h^3}{6} - \dots\\[1em] \implies& f(x+h) - f(x-h) = 2hf'(x) +
2\frac{h^3}{6} f'''(x) + 2\frac{h^5}{5!} f^{(5)}(x) + \dots \\ \implies&
g(h) \stackrel{def}{=} \frac{f(x+h) - f(x-h)}{2h} = f'(x) + h^2
\frac{f'''(x)}{3!} + h^4 \frac{f^{(5)}(x)}{5!} + \dots \\ \implies& g(h) =
f'(x) + \sum_{i=0}^{\infty} a_i h^{2+2i} = f'(x) + \mathcal{O}(h^2)
$$

where $\mathcal{O}(\cdot)$ denotes the Landau notation. Richardson's method can be
used to improve the error rate $\mathcal{O}(h^2)$.

## General case

In general Richardson's method considers sequences that can be written as:

$$
g(h) = L + \sum_{i=0}^{\infty} a_i h^{\theta +i \phi,}
$$

where $L \in \mathbb{R}$ denotes the limit of interest, $\theta$
the *base order of the approximation* and $\phi$ the *exponential step*. Allthough
Richardson's method works for general sequences, we are mostly interested in
the sequences arising when estimating derivatives.

### Example (contd.)

For standard derivative estimates we have

| Method         | $L$     | $\theta$ | $\phi$ |
| -------------- | ------- | -------- | ------ |
| forward diff.  | $f'(x)$ | 1        | 1      |
| backward diff. | $f'(x)$ | 1        | 1      |
| central diff.  | $f'(x)$ | 2        | 2      |

## Richardson Extrapolation

From the above table we see that, in general, central differences have a lower
approximation error $\mathcal{O}(h^2)$ than forward or backward differences
$\mathcal{O}(h)$.

> **Question**: Can we improve upon this further?

Let us evaluate $g$ at multiple values $h_0, h_1, h_2, \dots$, where it will
turn out to be useful to choose values $h, h/2,  h/4, h/8, \dots$ given some
prechosen $h > 0$. More generally $\{ h_n \}_n, h_n = h/2^n$ for $n
\in \mathbb{N}$. This allows us to write

$$
g(h) &= L + \sum_{i=0}^{\infty} a_i h^{\theta +i \phi}\\ g(h/2) &= L +
\sum_{i=0}^{\infty} a_i h^{\theta +i \phi} \frac{1}{2^{\theta +i \phi}}\\ g(h/4) &=
L + \sum_{i=0}^{\infty} a_i h^{\theta +i \phi} \frac{1}{4^{\theta +i \phi}}\\
&\vdots
$$

Now approximate the $g(h_n)$ by dropping all elements in the infinite sum after
$i=1$ and collect the approximation error using the term $\eta(h_n)$:

$$
g(h) &= \tilde{g}(h) + \eta(h) := L + \sum_{i=0}^{1} a_i h^{\theta +i \phi}
\\ g(h/2) &= \tilde{g}(h/2) + \eta(h/2) := L + \sum_{i=0}^{1} a_i h^{\theta
+i \phi} \frac{1}{2^{\theta +i \phi}}\\ g(h/4) &= \tilde{g}(h/4) +
\eta(h/4) := L + \sum_{i=0}^{1} a_i h^{\theta +i \phi} \frac{1}{4^{\theta
+i \phi}}\\ &\vdots
$$

Notice that we are now able to summarize the equations as

$$
\begin{bmatrix}
g(h) \\
g(h/2) \\
g(h/4)
\end{bmatrix}
=
 \begin{bmatrix}
  1 & h^\theta & h^{\theta + \phi} \\
  1 & {h^\theta}/{2^\theta} & {h^{\theta + \phi}}/{(2^{\theta + \phi})} \\
  1 & {h^\theta}/{4^\theta} & {h^{\theta + \phi}}/{(4^{\theta + \phi})} \\
  \end{bmatrix}
  \begin{bmatrix}
  L \\ a_0 \\ a_1
  \end{bmatrix}
+
  \begin{bmatrix}
  \eta (h)\\
  \eta (h/2) \\
  \eta (h/4)
  \end{bmatrix}
$$

which we write in shorthand notation as

$$
(\ast): \,\,\,
g = H
  \begin{bmatrix}
  L \\ a_0 \\ a_1
  \end{bmatrix}
+ \eta \,.
$$

From looking at equation ($\ast$) we see that an improved estimate of $L$
can be obtained by projecting $g$ onto $H$.

### Remark

To get a better intuition for ($\ast$) consider $H$ in more detail. For the
sake of clarity let $\theta = \phi = 2$.

$$
H =
\begin{bmatrix}
  1 & h^2 & h^4 \\
  1 & h^2/2^2 & h^4/2^4 \\
  1 & h^2/4^2 & h^4/4^4 \\
\end{bmatrix} =
\begin{bmatrix}
  1 & h^2 & h^4 \\
  1 & (h/2)^2 & (h/2)^4 \\
  1 & (h/4)^2 & (h/4)^4 \\
\end{bmatrix}
$$

Hence $H$ is a design matrix constructed from polynomial terms of degree
$0,2,4,\dots$ (in general: $0,\theta, \theta + \phi, \theta + 2\phi,\dots$)
evaluated at the observed points $h, h/2,h/4,h/8, \dots$.

In other words, dependant on the step-size of the derivative ($h$), we fit a
polynomial model to the derivative estimate and approximate the true derivative using
the fitted intercept.

The usual estimate is then given by $\hat{L} := e_1^T (H^T H)^{-1} H^T g$ which is
equal to $e_1^T H^{-1} g = \sum_{i} \{H^{-1}\}_{1,i} g_i$ in case $H$ is
regular.

## Did we improve the error rate?

Let us first consider the error function $\eta: h \to \eta (h)$ in more detail. We
see that

$$
\eta(h) = g(h) - \tilde{g}(h) = L + \sum_{i=0}^{\infty} a_i h^{\theta +i
\phi} - (L +  \sum_{i=0}^{1} a_i h^{\theta +i \phi}) = \sum_{i=2}^{\infty}
h^{\theta +i \phi} = \mathcal{O}(h^{\theta +2 \phi}) \,.
$$

Now consider the case where $H$ is regular (which happens here when $H$ is
quadratic). We then have, using ($\ast$)

$$
g = H
   \begin{bmatrix}
   L \\ a_0 \\ a_1
   \end{bmatrix}
+ \eta \implies H^{-1} g =
\begin{bmatrix}
  L \\ a_0 \\ a_1
\end{bmatrix}
+ H^{-1} \eta
$$

To get a better view on the error rate consider our ongoing example again.

### Example (contd.)

With

$$
H =
\begin{bmatrix}
  1 & h^2 & h^4 \\
  1 & (h/2)^2 & (h/2)^4 \\
  1 & (h/4)^2 & (h/4)^4 \\
\end{bmatrix}
$$

we get

$$
H^{-1} = \frac{1}{45}
 \begin{bmatrix}
    1       & -20      & 64\\
    -20/h^2 & 340/h^2  & -320/h^2\\
    64/h^4  & -320/h^4 & 256/h^4
 \end{bmatrix}
$$

Further, since for central differences $\theta = \phi = 2$ we have $\eta
(h_n) = \mathcal{O}(h^6)$ for all $n$ and thus:

$$
H^{-1} \eta = H^{-1}
\begin{bmatrix}
  \eta(h) \\
  \eta (h/2) \\
  \eta (h/4) \\
\end{bmatrix}
=
\begin{bmatrix}
  \mathcal{O}(h^6) \\
  \dots \\
  \dots \\
\end{bmatrix}
\implies \hat{L} = \{H^{-1} g \}_1 = L + \mathcal{O}(h^6)
$$

And so indeed we improved the error rate.
