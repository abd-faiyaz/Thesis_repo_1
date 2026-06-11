\REQUIRE Training APK transactions $\mathcal{T}$, labels $\mathbf{y}$, thresholds $\theta_{\mathrm{conf}}, \theta_{\mathrm{lift}}$
\ENSURE Selected permission set $S$ with $|S| \leq 40$

\STATE Rank permissions by PRNR malware--benign rate delta $\rightarrow$ list $P$
\STATE $P' \leftarrow$ filter $P$ by support bounds $[0.01, 0.95]$ on $\mathcal{T}$
\STATE $\mathcal{T}_m \leftarrow \{t \in \mathcal{T} : y(t) = \mathrm{malware}\}$
\STATE $R \leftarrow$ mine association rules on $\mathcal{T}_m$ with confidence $\geq \theta_{\mathrm{conf}}$, lift $\geq \theta_{\mathrm{lift}}$
\STATE $S \leftarrow$ permissions appearing in rule consequents of $R$
\IF{$|S| < 5$}
   \STATE Backfill $S$ from $P'$ until $|S| = 5$
\ENDIF
\STATE Truncate $S$ to at most 40 permissions, preserving PRNR order

\endinput
