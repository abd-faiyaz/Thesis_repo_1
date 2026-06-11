\REQUIRE PyTorch model $f_\theta$, ONNX session $g$, parity samples $\{(\mathbf{x}_j, p_j^{\mathrm{ref}})\}$, tolerance $\tau = 10^{-4}$
\ENSURE Parity report PASS or FAIL

\STATE $\delta_{\max} \leftarrow 0$
\FOR{each sample $j$}
   \STATE $p^{\mathrm{pt}} \leftarrow f_\theta(\mathbf{x}_j)$
   \STATE $p^{\mathrm{onnx}} \leftarrow g(\mathbf{x}_j)$
   \STATE $\delta_j \leftarrow |p^{\mathrm{pt}} - p^{\mathrm{onnx}}|$
   \STATE $\delta_{\max} \leftarrow \max(\delta_{\max}, \delta_j)$
\ENDFOR
\IF{$\delta_{\max} \leq \tau$}
   \RETURN PASS
\ELSE
   \RETURN FAIL
\ENDIF

\endinput
