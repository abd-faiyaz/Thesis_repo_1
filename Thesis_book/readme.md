Here’s how this repo is wired.

**Main Entry**
- Edit compile root: `Thesis Template UG/buetcseugthesis.tex:1`
- This file loads packages, starts the document, and controls chapter order using `\input{...}`.
- To add/remove/reorder chapters, edit the `\input{...}` lines in `Thesis Template UG/buetcseugthesis.tex:31`.

**Front Matter**
- Thesis title: `Thesis Template UG/parameters/thesistitle.txt:1`
- Thesis date: `Thesis Template UG/parameters/thesisdate.txt:1`
- Student names/IDs: `Thesis Template UG/parameters/students.txt:1`
- Supervisor: `Thesis Template UG/parameters/supervisor.txt:1`
- Abstract: `Thesis Template UG/buetcseugthesisabstract.tex:1`
- Acknowledgement: `Thesis Template UG/buetcseugthesisacknowledgement.tex:1`

Important: `Thesis Template UG/inputs/abstract.tex:1` and `Thesis Template UG/inputs/acknowledgement.tex:1` look like unused older/example files. The active files are `buetcseugthesisabstract.tex` and `buetcseugthesisacknowledgement.tex`, because the style file includes those directly.

**Main Chapters To Edit**
- Introduction: `Thesis Template UG/buetcseugthesisintroduction.tex:1`
- Background / related works: `Thesis Template UG/buetcseugthesisciteexamples.tex:1`
- Proposed methodology: `Thesis Template UG/buetcseugthesisproposedmethodology.tex:1`
- Experiments: `Thesis Template UG/buetcseugthesisexperiment.tex:1`
- Ethics / professional practice: `Thesis Template UG/buetcseugthesisethics.tex:1`
- Conclusion: `Thesis Template UG/buetcseugthesisconclusion.tex:1`

**References And Extras**
- Bibliography wrapper: `Thesis Template UG/buetcseugthesisbibliography.tex:1`
- BibTeX database: `Thesis Template UG/buetcseugthesis.bib:1`
- Algorithms appendix: `Thesis Template UG/buetcseugthesisalgorithms.tex:1`
- Code appendix: `Thesis Template UG/buetcseugthesiscodes.tex:1`
- Code files included in appendix: `Thesis Template UG/codes/fibonacci.c:1`, `Thesis Template UG/codes/salesa1.sql:1`
- Figures/images: `Thesis Template UG/figures/buetlogo.png`, `Thesis Template UG/figures/sample.jpg`

**Usually Do Not Edit**
- Template/style logic: `Thesis Template UG/buetcseugthesis.sty:1`
- Generated LaTeX build files: `.aux`, `.bbl`, `.blg`, `.toc`, `.lof`, `.lot`, `.log`, `.out`, `.fls`, `.fdb_latexmk`, `.idx`, `.ilg`, `.ind`, `.synctex`
- Existing compiled PDF unless you are rebuilding: `Thesis Template UG/buetcseugthesis.pdf`

**Practical Editing Rule**
- For thesis content, edit the chapter `.tex` files.
- For metadata, edit files under `Thesis Template UG/parameters/`.
- For citations, add entries to `Thesis Template UG/buetcseugthesis.bib:1` and cite them from chapter files.
- For structure/order, edit `Thesis Template UG/buetcseugthesis.tex:31`.