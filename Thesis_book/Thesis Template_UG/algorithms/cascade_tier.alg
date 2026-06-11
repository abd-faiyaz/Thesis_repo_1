\REQUIRE Tier scores $\{s_i\}$, weights $\{w_i\}$, thresholds $t_{\mathrm{low}}, t_{\mathrm{high}}$, conservative flag $c$
\ENSURE Exit verdict $\in \{\mathrm{BENIGN}, \mathrm{MALWARE}, \mathrm{CONTINUE}\}$

\STATE $\bar{s} \leftarrow \sum_i w_i s_i / \sum_i w_i$
\IF{$\bar{s} \leq t_{\mathrm{low}}$}
   \RETURN $\mathrm{BENIGN}$
\ENDIF
\IF{$\bar{s} \geq t_{\mathrm{high}}$}
   \RETURN $\mathrm{MALWARE}$
\ENDIF
\IF{$c = \mathrm{true}$ \AND $\exists i : s_i \geq t_{\mathrm{high}}^{(i)}$}
   \RETURN $\mathrm{MALWARE}$ \textit{(conservative OR)}
\ENDIF
\RETURN $\mathrm{CONTINUE}$

\endinput
