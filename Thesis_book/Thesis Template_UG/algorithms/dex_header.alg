\REQUIRE APK archive $A$, train-only min--max stats $(\mathbf{m}, \mathbf{M})$
\ENSURE 104-dimensional header vector $\mathbf{H}$

\STATE $\mathbf{H} \leftarrow \mathbf{0}^{104}$
\FOR{each \texttt{classes*.dex} entry $d$ in $A$}
   \STATE $\mathbf{r} \leftarrow$ raw bytes $d[8{:}112]$ \textit{(104 header fields)}
   \STATE $\mathbf{n}_d \leftarrow (\mathbf{r} - \mathbf{m}) / (\mathbf{M} - \mathbf{m} + \epsilon)$
   \STATE $\mathbf{H} \leftarrow \mathbf{H} + \mathbf{n}_d$ \textit{(multidex sum-pool)}
\ENDFOR
\RETURN $\mathbf{H}$

\endinput
