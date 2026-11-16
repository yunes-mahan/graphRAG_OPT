# Third-party notices

This repository's own code is licensed under Apache-2.0 (see `LICENSE`). The files below contain or are derived from
third-party material and remain under their original licences.

## 1. Microsoft GraphRAG 3.3.0: community-report prompt (MIT)

Files (verbatim copies of `COMMUNITY_REPORT_PROMPT` in `graphrag/prompts/index/community_report.py`, graphrag 3.3.0,
each preceded by a one-line source header):
- `artifacts/v16/community_report_prompt_3.3.0.txt`
- `revision/v16/community_report_prompt_3.3.0.txt`

Copyright (c) Microsoft Corporation. Licence, copied verbatim from the graphrag 3.3.0 LICENSE
(`graphrag-3.3.0.dist-info/licenses/LICENSE` of the PyPI distribution):

```
    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```

## 2. AI-Scientist-v3 (Sakana AI): `token_task/` (Apache-2.0)

- `token_task/capture.py`: copied from SakanaAI/AI-Scientist-v3 (`examples/*/capture.py`, commit df83138).
- `token_task/evaluate.py`: modified by Yunes Mahan, 2026, from SakanaAI/AI-Scientist-v3
  (`examples/circpack/evaluate.py`, commit df83138); the run harness is derived from it, the scoring is new.

Copyright 2026 Sakana AI. Licensed under the Apache License, Version 2.0; the full licence text is in
`token_task/LICENSE` (copied from the AI-Scientist-v3 repository; the repository has no NOTICE file).

## 3. Wikipedia-derived data: `artifacts/wiki/communities.json` (CC BY-SA 4.0)

`artifacts/wiki/communities.json` contains community labels and LLM-written summaries derived from the lead sections
of English Wikipedia articles (revisions at 1 January 2017-2025, retrieved with `wiki/wiki_crawl.py`).
© Wikipedia contributors. This file is distributed under the Creative Commons Attribution-ShareAlike 4.0
International licence (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/. The article texts themselves
are not redistributed; `wiki/wiki_crawl.py` re-crawls them.
