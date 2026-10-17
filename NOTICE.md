# Third-party content and its terms

The MIT licence in `LICENSE` covers the code written for this project. It does not cover the following, which keep
their own terms:

- `data/items/dass42_items.csv`: DASS-42 item wording and scale key, reproduced unmodified. The DASS website states that
  the DASS questionnaire is in the public domain and that the forms may be copied without restriction but may not be
  modified or sold (www.psy.unsw.edu.au/dass/). Cite the manual: Lovibond, S.H. & Lovibond, P.F. (1995). Manual for the
  Depression Anxiety Stress Scales (2nd ed.). Sydney: Psychology Foundation.
- `data/items/ipipneo300_items.csv`: IPIP item text, in the public domain (ipip.ori.org). Item key from Johnson's OSF
  repository (osf.io/tbmh5); Johnson, J. A. (2014), Journal of Research in Personality, 51, 78-89.
- `embed/vendor/vader_lexicon.txt`: VADER lexicon, MIT licence, Copyright (c) 2016 C.J. Hutto
  (`embed/vendor/VADER-LICENSE.txt`).
- `embed/out/`: embeddings of the public-domain item texts above and of synthetic fixture items, computed with
  BAAI/bge-large-en-v1.5 (MIT), sentence-transformers/all-mpnet-base-v2 (Apache-2.0) and Qwen/Qwen3-Embedding-0.6B
  (Apache-2.0); licences as listed on each Hugging Face model page, checked 2026-10-17.

No survey response data, and no statistic computed from survey responses, is included. A SHA-256 of the host file and a git object id (SHA-1) of the rebuilt DASS-42 correlation matrix are kept so a rebuild can be checked; neither discloses a value.
