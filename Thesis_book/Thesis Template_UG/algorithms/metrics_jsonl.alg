\REQUIRE Append-only file \texttt{all\_scan\_metrics.jsonl}
\ENSURE Legacy aggregate \texttt{all\_scan\_metrics.json}

\STATE $\mathcal{S} \leftarrow \emptyset$, $\mathcal{E} \leftarrow \emptyset$
\FOR{each non-empty line $\ell$ in JSONL}
   \STATE $r \leftarrow \mathrm{parse}(\ell)$
   \IF{$r.\mathrm{type} = \mathrm{session}$}
      \STATE $\mathcal{E} \leftarrow \mathcal{E} \cup \{r\}$
   \ELSIF{$r$ contains field \texttt{stages}}
      \STATE $\mathcal{S} \leftarrow \mathcal{S} \cup \{r\}$
   \ENDIF
\ENDFOR
\STATE $\mathit{device} \leftarrow$ device metadata from first record in $\mathcal{S} \cup \mathcal{E}$
\STATE Write $\{\mathit{device}, \mathit{scans}{:}\mathcal{S}, \mathit{sessions}{:}\mathcal{E}\}$ to JSON

\endinput
