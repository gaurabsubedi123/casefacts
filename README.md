# casefacts

Ask questions of an OCR'd case file and get answers with the page attached.

Everything runs on your machine. The documents never leave it, no request goes
to any network, and the models are the ones already sitting in your Ollama
install. That is not a feature list item — for medical records under a
protective order it is the only arrangement that is allowed.

It is the second half of [ocrtool](../ocr). ocrtool turns a stack of scans into
searchable text. This turns that text into answers you can check.

```
casefacts ask "what injuries were diagnosed, and when?" --in ~/Desktop/records
```

---

## The problem this is built around

A 7B model running on a laptop will sometimes state something the record does
not say, in exactly the same confident tone as everything it got right. In a
medical-legal file that is not an inconvenience. It is the whole risk.

So the model is never asked for prose. It is asked for a list of findings, and
**every finding must carry a verbatim quote from the page it cites**. Then
casefacts goes and checks each quote against the page it was supposedly taken
from, before you ever read it:

| Verdict | What it means |
|---|---|
| **verified** | The quote is on the page it cites, word for word. |
| **close** | On that page, allowing for OCR mangling the characters. |
| **joined** | Every word is on the page in that order, but not as one run of text — the model read across two columns of a layout-preserved page. True to the page; not usable as a quotation. |
| **wrong page** | The quote is real but the model named the wrong page. The citation is corrected and the finding kept. |
| **unverified** | The quote is nowhere in the pages that were retrieved. Shown anyway, marked, sorted to the bottom. |

The check costs no tokens and catches the failure that matters most. Nothing is
hidden from you: knowing the model made something up is more useful than not
seeing it.

Then there is the second check, which is the one that actually settles things.
Click any citation in the web interface and the **scanned page appears beside
the quote**, with the quoted words highlighted. Reading the answer is optional.
Checking it takes one click.

---

## Install

You need [ocrtool](../ocr) (for documents that are not OCR'd yet), Ollama, and
about 8 GB of disk for models.

```bash
cd ~/Desktop/casefacts
make install                          # or: uv venv && uv pip install -e ".[dev]"

ollama serve &                        # if it is not already running
ollama pull qwen2.5:7b-instruct       # answers questions
ollama pull nomic-embed-text          # the meaning half of the search
ollama pull medgemma:4b               # optional: a second opinion

.venv/bin/casefacts doctor            # checks all of the above
```

`doctor` tells you what is missing and the exact command to fix it.

---

## Using it

### Plug something in and ask about it

Point at anything. One PDF, one scan, a folder of them, or a folder ocrtool has
already been through. Whatever is not OCR'd yet gets OCR'd first, into a private
workspace under `~/.casefacts/ocr/`. **Nothing is ever written next to your
documents.**

```bash
# one file — OCR'd on the spot if it needs it, then read whole
casefacts ask "what did this report conclude?" --in ~/Desktop/MRI-report.pdf

# a whole folder
casefacts ask "list every provider who treated the patient" --in ~/Desktop/records

# an ocr-output folder ocrtool already produced — nothing to OCR, indexed in seconds
casefacts add /mnt/c/Users/you/Desktop/ocr-output
```

Once something is plugged in it stays indexed, so later questions skip straight
to the answer:

```bash
casefacts ask "when was the first complaint of back pain?"
casefacts ask "what did the ED find?" --file "Spring Valley"     # one document
casefacts ask "any imaging?" --folder ~/Desktop/records/imaging  # one folder
```

Plugging the same path in again re-reads only what changed. ocrtool keeps its
own ledger, so nothing is OCR'd twice either.

### One file, read whole — the "attach a document and ask" case

When a question is scoped to a single document small enough to fit in the
model's context, casefacts **skips retrieval entirely and puts the whole
document in the prompt**. For a twenty-page discharge summary, searching it
could only lose information.

That happens automatically. `--whole` forces it; the answer is then labelled
`whole document` rather than `search`, so you always know which one you got.

Above about 60,000 characters — measured at roughly 33 pages of this claim
file, which averages 1,800 characters a page — it falls back to searching
within that document, because the alternative is a prompt that silently gets
truncated.

### The chronology

For an injury case this is the document everything else is built on. Every page
read once, in order, for anything with a date on it:

```bash
casefacts chronology --csv timeline.csv
```

```
2018-07-06   Emergency department admission, motor vehicle accident   — Spring Valley Hospital
             58 Marquis Arbauch subpeona p.65 (GEICO000065)
2018-12-03   Lumbar Transforaminal Epidural Injection                 — Jones Physical Therapy
             58 Marquis Arbauch subpeona p.183 (GEICO000183)

Gaps of 30+ days with no record:
   281 days   2018-12-10 -> 2019-09-17

These are what a defence examiner will call a break in treatment.
```

Every row carries the page it came from and the quote that establishes it, and
every date is checked against the page before the row is kept — a chronology
sorted on an invented date is worse than no chronology.

It is resumable. Interrupt it with Ctrl-C, run it again, and it picks up where
it stopped. Pages with no date on them never reach a model at all, which is
what makes sweeping a 2,000-page file affordable.

### Comparing models on your own records

There is no benchmark for "reads this claim file correctly". So run several and
look at what they do differently:

```bash
casefacts ask "what was the mechanism of injury?" --all-models
```

```
Pages every model cited: p.65, p.128
Agreement on pages: 67%
  only medgemma:4b: p.292

The pages every model landed on are the ones to read first.
```

Two models reaching the same page from different wordings is the strongest
signal this tool can give you that the page is the right one.

---

## Which model to use

**Start with `qwen2.5:7b-instruct`, and do not reach for a medical model
first.** That is the opposite of the obvious advice, so here is the reasoning.

Meditron, BioMistral and OpenBioLLM are fine-tuned to answer board-exam
questions *from memory*. That is precisely the behaviour you do not want here,
where the model must read only what is on the page and refuse to fill gaps. They
are also weaker at following a citation format, and they now trail general
models like Qwen on aggregate medical benchmarks anyway.

For extraction-with-citations, **retrieval quality dominates medical domain
knowledge**. The clinical facts are in the document, not in the weights.

`medgemma:4b` is the medical model worth having — it is Google's, current, and
it genuinely knows the abbreviations and vocabulary the general models guess at.
Use it as a **second opinion**, not as the base. Being medically tuned also makes
it readier to supply textbook knowledge that is not in the record, so check its
citations harder, not less.

`casefacts models` prints all of this next to what you have installed.

### A note on VRAM

This machine has an RTX 4070 Laptop with 8 GB of VRAM, and Ollama runs these
models **100% on the GPU** — which is why a question comes back in seconds
rather than minutes.

That 8 GB is the real constraint, because the KV cache lives there next to the
weights. `qwen2.5:7b-instruct` loads at 5.0 GB with an 8k context and leaves
room; `qwen3:8b` and `medgemma:4b` both fit. A 27B model does not, so MedGemma's
27B variant is out of reach here and the 4B is the right one.

Ask for a context window bigger than the remaining VRAM and Ollama quietly
spills the model into system RAM, where the same question takes minutes.
Nothing reports this but the clock, which is why `MAX_NUM_CTX` is capped.

Two 5 GB models cannot be resident at once, so comparisons run sequentially and
the first question after a model switch pays the load time.

---

## How it works

### Reading

ocrtool's `.txt` files already carry a marker at the top of every page:

```
----- page 14 (ocr) -----
```

So page numbers are not guessed from form feeds or inferred from headers — they
are stated. And its `.json` carries, per page, the OCR confidence and the path
to a JPEG of that page. Those two facts are what make a citation here checkable
by eye rather than merely plausible.

Production stamps (`GEICO000472`) are pulled out too, but only when they form a
real **series**: one prefix, running across most of the document, with the number
changing page to page. Pattern-matching alone would call `Macon, GA 31294` a
Bates stamp; requiring a series does not.

### Chunking

**A chunk never crosses a page boundary.** It would retrieve slightly better if
it could, and it is still not allowed, because every answer has to be checkable
against a single page image. A citation spanning two pages is half right, and
half right is the worst kind of wrong in a medical record.

### Search

Both halves run and their rankings are fused:

- **FTS5 keyword search** finds `GEICO000472`, `CPT 99213`, `M25.512`, and a
  surname — the exact strings a medical-legal file is full of, where being close
  in meaning is worth nothing.
- **Vector search** finds "what did the orthopedist say about the shoulder" when
  the note says "L glenohumeral joint".

Neither alone is good enough. They are combined by reciprocal rank fusion rather
than a weighted sum, because bm25 and cosine similarity are not on the same
scale and any weighting between them would be a number invented to look
principled.

A production stamp typed into a question is looked up directly and pinned to the
top, because no amount of semantic similarity substitutes for looking it up.

Dense search always returns its nearest neighbours, however far away they are.
So if **none** of a question's own words appear anywhere in the file, the answer
says so rather than presenting the closest text as though it were relevant.

### Why there is no vector database here

A case file is big for a person and small for a computer. A 358-page production
is 751 chunks. A 10,000-page file would be about 21,000 — at 768 dimensions,
64 MB of float32, searched by one numpy dot product in milliseconds. A vector
store would add a dependency, a process, and a second copy of the evidence, and
would not be measurably faster until this held more pages than a firm produces
in a year.

---

## Measured on this machine

WSL2, RTX 4070 Laptop (8 GB VRAM), on the 358-page GEICO claim file:

| | |
|---|---|
| Indexing an already-OCR'd folder | 359 pages, 751 chunks, **9 seconds** |
| Embedding throughput | ~90 chunks/second |
| One question, hybrid search | **6–13 seconds** |
| One question, whole small document | ~7 seconds |
| Chronology sweep, all 358 pages | **15 minutes** — 246 pages read at 3.7s each, 112 skipped for having no date and costing nothing |
| What the sweep produced | 269 events → 230 rows after collapsing repeats, spanning 2018-07-06 to 2019-09-17, and 2 treatment gaps (34 and 196 days) |
| What the checker caught there | 223 events verified; 32 carried a date not on their page and 14 a quote that was not; all 46 held back from the timeline |
| Duplicate detection | 7 of 9 files caught as copies — embedded once, not four times |
| Test suite | 98 tests, ~3 seconds, no model needed |

A 2,000-page file is therefore roughly a minute to index and an hour and a
half to sweep for a chronology, once — and it is resumable, so that hour can be
taken in pieces.

---

## The web interface

```bash
casefacts ui              # http://127.0.0.1:5002
```

Port 5002 because ocrtool's interface is on 5001 and the litigation wiki is on
5000.

Two columns, and that is the entire design: answers on the left, the scanned
page on the right. Checking a citation never means losing your place.

- Plug in a path from the top bar, and watch OCR and indexing progress live.
- Ask, with a scope (everything / one folder / one document) and any set of
  models ticked. Tick more than one to compare them side by side.
- Every finding shows its verdict badge, its quote, and a citation button.
- Click the citation → the page image and page text appear on the right, with
  the quoted words highlighted.
- The chronology tab builds the timeline, shows the gaps inline, and downloads
  a CSV.

Server-rendered HTML and plain JavaScript. No build step, no bundler, nothing
loaded from a CDN — a tool that reads medical records should not be reaching out
to the network to render a page.

---

## Where things are kept

```
~/.casefacts/index.db          the index: pages, chunks, vectors, events
~/.casefacts/ocr/<name>-<id>/  OCR of documents that were not OCR'd already
~/.casefacts/config.json       your defaults
```

The index lives on the Linux filesystem rather than under `/mnt/c` on purpose:
SQLite on a drvfs mount pays a syscall round trip per page fetch, and the same
index on ext4 is faster by more than an order of magnitude. Your documents stay
where they are; only the derived index moves.

`casefacts forget <path>` drops everything that came from one path. Your files
are never touched.

---

## What it does not do

- **It is not advice, and it is not a substitute for reading the file.** It is a
  finding aid that points you at pages and shows you the page.
- **It does not decide what is true.** Where two records disagree it is told to
  report both and say they disagree.
- **It cannot see what the OCR did not read.** A page at 40% confidence is
  flagged in every answer that rests on it — go and look at that page.
- **It does not do medical reasoning.** Nothing here diagnoses, stages, rates
  impairment, or apportions causation.
- **No open medical model is validated for clinical decisions**, MedGemma's own
  documentation included. This tool's output is a citation to a page in a
  record, and that is all it claims to be.

---

## Commands

```
casefacts add <path>          plug in a file or folder and index it
casefacts ask "<question>"    ask; --in / --file / --folder to scope it
                              --model, --compare a,b, --all-models, --whole
casefacts chronology          build the dated timeline; --csv to export
casefacts search "<terms>"    find pages without asking a model anything
casefacts page <doc> <n>      print one page as it was read
casefacts documents           what is indexed, and which copies are duplicates
casefacts sources             what has been plugged in
casefacts forget <path>       drop one source from the index
casefacts models              what is installed, and what each is good for
casefacts doctor              check Ollama, the models, ocrtool and the index
casefacts ui                  the web interface on port 5002
casefacts config              show or set the defaults
```

Every command takes `--db` to work against a different index, which is how you
keep two matters apart.

---

MIT. Local only, and it stays that way.
