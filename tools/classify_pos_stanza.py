#!/usr/bin/env python3
"""Experimental: same job as `tools/classify_pos.py`, but the POS rule is a
Russian parse instead of a suffix list.

`classify_pos.py` guesses from endings: -ть/-ти is a verb, -ый/-ий/-ой/-ая is an
adjective, anything else in the gloss is the head noun. That is fast and needs
nothing installed, but it is a lexicalised guess, and it breaks exactly where a
gloss is not one word with a canonical ending:

    "очень вкусный"        -> n       ('очень' is neither adj nor verb)
    "самый большой"        -> n       (same)
    "печь"                 -> n       (verb and noun share the ending)
    "было холодно"         -> n       ('было' is the first non-adj/verb word)
    "вид, выражение"       -> n       right, but for the wrong reason

Here stanza's Russian tokenizer + POS tagger + dependency parser (Universal
Dependencies) does the analysis: the gloss is parsed, the syntactic root of the
phrase is taken as the head, and the head's UPOS/features decide the class. That
also recovers things the suffix rule cannot see at all -- animacy (the lexicon
splits N-AN from N-NN), proper names (Np), numerals, pronouns, adverbs.

This file is deliberately standalone and nothing imports it: run it side by side
with the shipped classifier and look at the disagreements before deciding
whether to adopt it. It is not wired into the Makefile, the tests, or the
lexicon build.

    python tools/classify_pos_stanza.py                  # sanity tests
    python tools/classify_pos_stanza.py "многоэтажный дом" "самый большой"
    python tools/classify_pos_stanza.py --compare REPO/../dictionary.txt
    python tools/classify_pos_stanza.py --compare glosses.csv --column gloss
    python tools/classify_pos_stanza.py --gold .tmp/gold_pos.csv   # accuracy

Requirements (optional dependency, not added to requirements*.txt):
    python -m pip install --target pylibs stanza
    # torch is pulled in with it; the Russian model (~50 MB) is downloaded on
    # first use into pylibs/stanza_resources (or $DUNGA_STANZA_RESOURCES).

Cost, for the decision: the suffix rule runs over 10k glosses in well under a
second; stanza adds torch to the environment, downloads a model once, and takes
on the order of a minute for the same 10k glosses on CPU.

Conventions worth knowing before reading the code:
  * `classify()` returns the same small tag vocabulary as the shipped module
    (n / adj / vblex / adv, plus np / num / prn / post / cnj / part / other when
    the parse is unambiguous about a class the old module never produced), so
    the two can be compared word for word.
  * `classify_full()` returns the actual POS tag used in stems/*.lexc
    (N-NN, N-AN, Np, Adj, V, Adv, Num, Prn, Post, CnjCoo ...); the tone prefix
    (T1-/T2-/T3-) is not this module's business.
  * Glosses are multi-sense more often than not (21% carry a ';'): each sense is
    parsed, and the first sense that parses decides, mirroring the dictionary's
    first-meaning-first order. `Decision.mixed` flags the cases where the senses
    disagreed, which are worth a human look.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
from pathlib import Path

MODEL_LANG = "ru"
PROCESSORS = "tokenize,pos,depparse"

# ---------------------------------------------------------------- bootstrap --

_HERE = Path(__file__).resolve()
_PYLIBS = [
    Path(p) for p in (
        os.environ.get("DUNGA_PYLIBS", ""),
        _HERE.parents[3] / "pylibs",  # <workspace>/pylibs  (tools/ -> repo -> workspace)
        _HERE.parents[2] / "pylibs",
        _HERE.parents[1] / "pylibs",
    ) if p
]


def _stanza_importable():
    return importlib.util.find_spec("stanza") is not None


def _bootstrap():
    """Pick up a workspace-local install (`pip install --target pylibs stanza`)."""
    if _stanza_importable():
        return True
    for d in _PYLIBS:
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return _stanza_importable()


def _resources_dir():
    env = os.environ.get("DUNGA_STANZA_RESOURCES")
    if env:
        return Path(env)
    for d in _PYLIBS:
        if d.is_dir():
            return d / "stanza_resources"
    return Path.home() / "stanza_resources"


def _model_present(d):
    ru = Path(d) / MODEL_LANG
    return all((ru / p).is_dir() for p in ("tokenize", "pos", "depparse"))


_NLP = None


def get_pipeline(download=True):
    """Build (once) the Russian stanza pipeline. Raises SystemExit with a hint."""
    global _NLP
    if _NLP is not None:
        return _NLP
    if not _bootstrap():
        raise SystemExit(
            "stanza is not installed.\n"
            "  python -m pip install --target pylibs stanza\n"
            "then re-run; the directory is found automatically (or set DUNGA_PYLIBS).")
    import stanza
    res = _resources_dir()
    if not _model_present(res):
        if not download:
            raise SystemExit(f"no Russian model in {res}; run once with --download")
        res.mkdir(parents=True, exist_ok=True)
        stanza.download(MODEL_LANG, dir=str(res), processors=PROCESSORS, verbose=False)
    _NLP = stanza.Pipeline(MODEL_LANG, processors=PROCESSORS, dir=str(res),
                           download_method=None, verbose=False, use_gpu=False)
    return _NLP


# ------------------------------------------------------------------- tags ----

# UPOS -> lexc POS. Nouns need features, so they are handled separately.
_LEXC = {
    "PROPN": "Np", "ADJ": "Adj", "VERB": "V", "AUX": "V", "ADV": "Adv",
    "NUM": "Num", "DET": "Prn", "PRON": "Prn", "ADP": "Post",
    "CCONJ": "CnjCoo", "SCONJ": "CnjSub", "PART": "Part", "INTJ": "Ij",
}

# lexc POS -> the small vocabulary `classify_pos.classify()` speaks.
_MINIMAL = {
    "N-NN": "n", "N-AN": "n", "Np": "np", "Adj": "adj", "AdjDeg": "adj",
    "V": "vblex", "Adv": "adv", "Num": "num", "Prn": "prn", "Post": "post",
    "CnjCoo": "cnj", "CnjSub": "cnj", "Part": "part", "Ij": "ij", "Loc": "loc",
}

# For the A/B against the suffix rule, whose whole vocabulary is n/adj/vblex/adv.
_COARSE = {"np": "n", "loc": "adv"}

# Where an unparsed gloss has to fall back, the suffix rule's four classes have
# to be spelled in lexc terms again.
_FROM_MINIMAL = {"n": "N-NN", "adj": "Adj", "vblex": "V", "adv": "Adv"}

# A participle used as a gloss ("испечённый", "устаревший") names an adjective-
# like property and the lexicon files such lemmas under Adj. Set to "V" to keep
# them verbal instead.
PARTICIPLE_TAG = "Adj"

_SKIP_UPOS = {"PUNCT", "SYM", "X"}
_CLIMB_UPOS = {"AUX", "PART", "CCONJ", "SCONJ"}  # function words with content children

# ------------------------------------------------------------------ input ----

# OCR markers for the special Dungan letters (ХcyrZhedesc -> җ etc.) leak into
# glosses from the PDF text layer. Tone codes leak too. Neither is Russian.
_CYR_MARKER = re.compile(r"[ХXхx]?cyr[A-Za-z0-9]+")
_ROMAN_CODE = re.compile(r"(?<![\wА-Яа-яЁё])[IVX]{1,4}(?:-[IVX]{1,4})+(?![\wА-Яа-яЁё])")
_NUMBERED = re.compile(r"^\s*(?:\d+|\([0-9а-я]\)|[а-я]\))\s*[.)]?\s+", re.I)
_XREF = re.compile(r"^\s*(?:см|ср|сравни|то\s+же)\b", re.I)
_HEADWORD_LEAK = re.compile(r"\s[А-ЯЁӘҖҢҮЎ]{3,}(?:\s+[IVX]+(?:-[IVX]+)+\b)")  # "... ГА (ГЕКТАР) I"
_CYRILLIC = re.compile(r"[а-яёәҗңүў]", re.I)


def _clean(text):
    text = _CYR_MARKER.sub(" ", text)
    text = _ROMAN_CODE.sub(" ", text)
    text = _HEADWORD_LEAK.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _senses(gloss):
    """Split a gloss into senses ('1. ком; 2. шишка' -> ['ком', 'шишка, опухоль'])."""
    out = []
    for chunk in re.split(r"\s*;\s*", _clean(gloss)):
        chunk = _NUMBERED.sub("", chunk).strip(" .,:;—-")
        if not chunk or _XREF.match(chunk) or not _CYRILLIC.search(chunk):
            continue
        out.append(chunk)
    return out


# ------------------------------------------------------------------ parse ----


def _feats(word):
    d = {}
    for part in (word.feats or "").split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d


def _tag_of(word):
    upos = word.upos
    feats = _feats(word)
    if upos == "NOUN":
        return "N-AN" if feats.get("Animacy") == "Anim" else "N-NN"
    if upos == "ADJ" and feats.get("Degree") in ("Cmp", "Sup"):
        return "AdjDeg"
    if upos == "VERB" and feats.get("VerbForm") == "Part":
        return PARTICIPLE_TAG
    return _LEXC.get(upos, "")


def _head_of(sentence):
    """The syntactic root, walked down out of copulas/particles to a content word."""
    words = [w for w in sentence.words if w.upos not in _SKIP_UPOS]
    if not words:
        return None
    head = next((w for w in words if w.head == 0), words[0])
    seen = set()
    while head.upos in _CLIMB_UPOS and head.id not in seen:
        seen.add(head.id)
        kids = [c for c in words if c.head == head.id]
        if not kids:
            break
        kids.sort(key=lambda c: (c.upos in _CLIMB_UPOS or c.upos in ("DET", "ADP"), c.id))
        head = kids[0]
    return head


class Decision:
    """What the parse decided about one gloss, with the evidence kept around."""

    __slots__ = ("gloss", "tag", "minimal", "head", "upos", "deprel", "feats",
                 "sense", "senses", "mixed")

    def __init__(self, gloss, tag, minimal, head, upos, deprel, feats, sense, senses):
        self.gloss = gloss
        self.tag = tag
        self.minimal = minimal
        self.head = head
        self.upos = upos
        self.deprel = deprel
        self.feats = feats
        self.sense = sense
        self.senses = senses
        classes = {m for _, m in senses if m}
        self.mixed = len(classes) > 1

    def evidence(self):
        feats = ",".join(f"{k}={v}" for k, v in sorted(self.feats.items()))
        return f"{self.head}/{self.upos}" + (f" [{feats}]" if feats else "")

    def __repr__(self):
        return f"<Decision {self.gloss!r} -> {self.tag} ({self.minimal}) via {self.evidence()}>"


def _parse(texts):
    nlp = get_pipeline()
    if hasattr(nlp, "bulk_process"):
        docs = nlp.bulk_process(texts)
    else:  # older stanza
        docs = [nlp(t) for t in texts]
    heads = []
    for doc in docs:
        heads.append(_head_of(doc.sentences[0]) if doc.sentences else None)
    return heads


def analyse_many(glosses):
    """Classify a batch: one pipeline pass over every sense of every gloss."""
    sense_lists = [_senses(g) for g in glosses]
    flat = [s for senses in sense_lists for s in senses]
    heads = dict(zip(flat, _parse(flat))) if flat else {}
    out = []
    for gloss, senses in zip(glosses, sense_lists):
        decided = []
        for sense in senses:
            head = heads.get(sense)
            if head is None:
                continue
            tag = _tag_of(head)
            if not tag:
                continue
            decided.append((sense, tag, _MINIMAL.get(tag, "other"), head))
        if not decided:
            out.append(None)
            continue
        sense, tag, minimal, head = decided[0]
        out.append(Decision(gloss, tag, minimal, head.text, head.upos, head.deprel,
                            _feats(head), sense, [(s, m) for s, _, m, _ in decided]))
    return out


def analyse(gloss):
    return analyse_many([gloss])[0]


# ---------------------------------------------------------- regex fallback --

_REGEX_CLASSIFY = None


def classify_regex(gloss):
    """The shipped suffix heuristic, loaded from the sibling module."""
    global _REGEX_CLASSIFY
    if _REGEX_CLASSIFY is None:
        path = _HERE.with_name("classify_pos.py")
        spec = importlib.util.spec_from_file_location("_dunga_classify_pos", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _REGEX_CLASSIFY = mod.classify
    return _REGEX_CLASSIFY(gloss)


def classify(gloss, fallback=True):
    """Drop-in for `classify_pos.classify()`: n / adj / vblex / adv (+ np, num...)."""
    d = analyse(gloss)
    if d is None:
        return classify_regex(gloss) if fallback else "other"
    return d.minimal


def classify_full(gloss, fallback=True):
    """The stems/*.lexc POS tag, e.g. N-AN or N-NN (tone prefix not included)."""
    d = analyse(gloss)
    if d is None:
        if not fallback:
            return ""
        return _FROM_MINIMAL.get(classify_regex(gloss), "N-NN")
    return d.tag


def coarse(tag):
    """Collapse a tag to the four classes the suffix rule can produce."""
    return _COARSE.get(tag, tag)


# ----------------------------------------------------------------- report ----


def _load_glosses(path, column=None, unique=False, header=True):
    """Glosses from a CSV (--column), a heads+tone+gloss dump, or plain lines."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    glosses = []
    if column is not None:
        rows = list(csv.reader(raw))
        if not rows:
            return []
        if column.isdigit():
            idx = int(column)
        else:
            idx = rows[0].index(column)  # KeyError -> "column not in ...", clear enough
        start = 1 if header else 0
        glosses = [r[idx] for r in rows[start:] if len(r) > idx]
    elif p.suffix.lower() == ".csv":
        raise SystemExit("CSV input needs --column NAME_OR_INDEX (e.g. --column gloss)")
    else:
        rx = re.compile(r"^([А-ЯЁӘҖҢҮЎ]+)\s+[IVX]+(?:-[IVX]+)+\s+(.+)$")
        for line in raw:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = rx.match(line)
            glosses.append((m.group(2) if m else line).strip())
    glosses = [g for g in glosses if g.strip()]
    if unique:
        seen, dedup = set(), []
        for g in glosses:
            if g not in seen:
                seen.add(g)
                dedup.append(g)
        glosses = dedup
    return glosses


def run_compare(glosses, show=15, out=None):
    """A/B the suffix rule against the parse over real glosses."""
    print(f"glosses: {len(glosses)}")
    decisions = analyse_many(glosses)
    rows, agree, skipped = [], 0, 0
    pairs = {}
    for gloss, d in zip(glosses, decisions):
        if d is None:
            skipped += 1
            continue
        base = classify_regex(gloss)
        a, b = coarse(base), coarse(d.minimal)
        agree += a == b
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
        rows.append((gloss, base, d.minimal, d.tag, d.evidence(), a == b))
    total = len(rows)
    if not total:
        print("nothing could be parsed")
        return rows
    print(f"compared: {total}   unparsed: {skipped}")
    print(f"agreement: {agree}/{total} = {agree / total:.1%}")
    print("\ndisagreements (suffix rule -> parse):")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if a != b:
            print(f"  {n:5d}  {a} -> {b}")
    if show:
        print(f"\nfirst {show} differing glosses:")
        shown = 0
        for gloss, base, minimal, tag, ev, same in rows:
            if same:
                continue
            print(f"  {gloss[:60]:60s} | {base:5s} | {minimal:5s} {tag:6s} | {ev}")
            shown += 1
            if shown >= show:
                break
    if out:
        with open(out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gloss", "suffix_rule", "stanza", "lexc_tag", "evidence", "agree"])
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")
    return rows


def run_gold(path, column=None):
    """Accuracy of both classifiers against hand-labelled coarse classes."""
    p = Path(path)
    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if "expected" not in rows[0]:
        raise SystemExit(f"{path} needs columns: gloss,expected")
    glosses = [r["gloss"] for r in rows]
    expected = [coarse(r["expected"].strip().lower()) for r in rows]
    decisions = analyse_many(glosses)
    hits = {"suffix_rule": 0, "stanza": 0}
    misses = []
    for gloss, want, d in zip(glosses, expected, decisions):
        got_rule = coarse(classify_regex(gloss))
        got_stanza = coarse(d.minimal) if d else "<unparsed>"
        hits["suffix_rule"] += got_rule == want
        hits["stanza"] += got_stanza == want
        if got_rule != want or got_stanza != want:
            ev = d.evidence() if d else "-"
            misses.append((gloss, want, got_rule, got_stanza, ev))
    n = len(glosses)
    print(f"gold glosses: {n}")
    for name in ("suffix_rule", "stanza"):
        print(f"  {name:12s} {hits[name]:3d}/{n} = {hits[name] / n:.1%}")
    print("\nwhere either missed:")
    for gloss, want, gr, gs, ev in misses:
        flag = ("both " if gr != want and gs != want else
                "rule " if gr != want else "stanza")
        print(f"  [{flag}] {gloss[:52]:52s} want {want:6s} rule {gr:6s} stanza {gs:6s} | {ev}")
    return hits


SELF_TESTS = [
    # (gloss, expected coarse class) -- the shipped module's own test set first.
    ("високосный год", "n"), ("литературный язык", "n"), ("многоэтажный дом", "n"),
    ("вкусный", "adj"), ("танцевать плясать", "vblex"), ("голая степь", "n"),
    ("маленький", "adj"), ("тёща", "n"),
    # Cases the suffix list gets wrong or cannot see.
    ("очень вкусный", "adj"), ("самый большой", "adj"), ("печь хлеб", "vblex"),
    ("вид, выражение", "n"), ("наряд, одежда", "n"), ("отступить, падать", "vblex"),
    ("предвечерние сумерки", "n"), ("поскольку", "cnj"), ("внизу", "adv"),
    ("пять", "num"), ("он", "prn"), ("в течение", "post"),
]


def run_self_tests():
    glosses = [g for g, _ in SELF_TESTS]
    decisions = analyse_many(glosses)
    bad = 0
    for (gloss, want), d in zip(SELF_TESTS, decisions):
        got = coarse(d.minimal) if d else "<unparsed>"
        ok = got == want
        bad += not ok
        print(f"  {'OK ' if ok else 'XX '} {gloss!r} -> {got} (want {want})"
              f"{'' if ok else '  via ' + (d.evidence() if d else 'nothing parsed')}")
    print(f"\n{len(SELF_TESTS) - bad}/{len(SELF_TESTS)} passed")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stanza-backed POS classifier for Yanshansin Russian glosses "
                    "(experiment; see module docstring).")
    ap.add_argument("glosses", nargs="*", help="glosses to classify")
    ap.add_argument("--compare", metavar="FILE",
                    help="A/B the suffix rule against the parse over a gloss list")
    ap.add_argument("--gold", metavar="CSV", help="gloss,expected CSV for accuracy")
    ap.add_argument("--column", help="gloss column for --compare/--gold CSV input")
    ap.add_argument("--no-header", action="store_true",
                    help="the CSV given to --column/--gold has no header row")
    ap.add_argument("--limit", type=int, help="cap the number of glosses")
    ap.add_argument("--unique", action="store_true", help="deduplicate glosses")
    ap.add_argument("--show", type=int, default=15, help="disagreements to print")
    ap.add_argument("--out", metavar="CSV", help="write --compare rows here")
    ap.add_argument("--tagset", choices=("minimal", "lexc"), default="minimal")
    ap.add_argument("--no-fallback", action="store_true",
                    help="do not fall back to the suffix rule when a gloss is unparsed")
    ap.add_argument("--download", action="store_true",
                    help="allow downloading the Russian model if it is missing")
    ap.add_argument("--self-test", action="store_true", help="run the built-in pairs")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Dungan letters

    get_pipeline(download=args.download or args.compare or args.gold or args.glosses)

    if args.self_test or not (args.compare or args.gold or args.glosses):
        return 1 if run_self_tests() else 0

    if args.gold:
        run_gold(args.gold, args.column)
        return 0

    if args.compare:
        glosses = _load_glosses(args.compare, args.column, args.unique,
                                header=not args.no_header)
        if args.limit:
            glosses = glosses[:args.limit]
        run_compare(glosses, show=args.show, out=args.out)
        return 0

    decisions = analyse_many(args.glosses)
    for gloss, d in zip(args.glosses, decisions):
        if d is None:
            print(f"{gloss!r:40s} -> unparsed  (suffix rule would say "
                  f"{classify_regex(gloss)})")
            continue
        tag = d.tag if args.tagset == "lexc" else d.minimal
        print(f"{gloss!r:40s} -> {tag:6s} via {d.evidence()}"
              f"  [{d.sense!r}{'  MIXED' if d.mixed else ''}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
