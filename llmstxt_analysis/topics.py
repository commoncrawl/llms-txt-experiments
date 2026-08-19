"""Track F — topics via LDA, with manually assigned human-readable names.

LDA is chosen for cost: a bag-of-words model over a 2 KB prose excerpt of the
human-authored subset fits in minutes on one core, versus hours for embedding
based clustering. Topic *names* are not machine-generated: ``topics fit`` dumps
top terms and representative documents, a human writes ``topic_names.yaml``, and
``topics assign`` bakes those names into the report input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from .derive import add_derived, load_features

# Scripts without whitespace word boundaries need a segmenter; out of scope.
UNSUPPORTED_SCRIPTS = {"jpn", "zho", "kor", "tha", "lao", "mya", "khm", "bod"}

DEFAULT_N_TOPICS = 20
DEFAULT_MIN_DOCS = 3000
TOP_TERMS = 15
N_EXAMPLES = 5

# scikit-learn ships an English stop-word list only. Without one for the other
# languages the top terms are almost entirely pronouns and auxiliaries, which
# makes the topics unnameable. These are compact function-word lists — closed
# word classes, no domain terms — for the languages large enough to model.
STOPWORDS: dict[str, set[str]] = {
    "deu": set("""aber alle allem allen aller alles als also andere anderen auch auf aus바 bei
        beim bereits bin bis bist damit dann das dass dazu dein deine dem den denn der des dessen
        deshalb dich die dies diese diesem diesen dieser dieses dir doch dort durch ein eine einem
        einen einer eines er es etwa etwas euch euer eure für gegen gibt hab habe haben hat hatte
        hatten hier hin ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres immer inbesondere ins
        ist jede jedem jeden jeder jedes jetzt kann kannst kein keine keinen können könnt lassen
        machen man mehr mein meine mich mir mit muss müssen nach nicht noch nun nur oder ohne über
        schon sehr sein seine seinem seinen seiner seit sich sie sind sowie soll sollen sondern
        über uns unser unsere unserem unseren unserer vom von vor wann war waren warum was weil
        weiter welche wenn wer werden wie wir wird wirst wo wollen wurde wurden zum zur zwischen
        alle darauf dabei dafür dadurch daher darüber deren dessen ebenfalls etc jedoch nämlich
        sowohl bzw ggf ihrerseits inkl""".split()),
    "fra": set("""alors and aux avec avoir avons ce ces cet cette ceux chaque comme dans des donc
        dont elle elles est etre eux fait faire font ils jusqu leur leurs mais mes moins mon nos
        notre nous ont par pas peut peuvent plus pour pourquoi quand que quel quelle quelles quels
        qui quoi sans ses soit sommes son sont sur tous tout toute toutes très une vos votre vous
        aussi autre autres avait avez bien car cela celle celui depuis deux doit encore entre etait
        etaient etes ete meme non ou puis sera seront si sous toujours vers etc afin ainsi lors
        selon chez peu doivent pourra""".split()),
    "spa": set("""algunas algunos ante antes aunque cada como con contra cual cuando desde donde
        dos ella ellas ellos entre era eran esta estas este estos fue fueron hace hacer hasta las
        les lo los más mientras muy nos nuestra nuestro otra otro para pero por porque puede pueden
        que quien saber ser si sin sobre son sus tambien tiene tienen todo todos una uno unos ver
        ya sea esto esa ese estar están hay""".split()),
    "por": set("""ainda alem algumas alguns ante antes apenas apos ate cada como com contra das
        dele deles depois desde dois dos ela elas ele eles entre era eram essa esse esta estas este
        estes foi foram isso isto mais mas mesmo muito nao nas nem nos nossa nosso ou para pela
        pelo pelos por porque qual quando que quem sao sem ser seu seus sobre sua suas tambem tem
        ter todos uma umas uns voce""".split()),
    "ita": set("""alcuni alla alle allo anche ancora avere caso che chi cioe come con cui dal dei
        del della delle dello dove due essere fare gli hanno inoltre loro ma mentre molto nel nella
        nelle noi non oltre ogni per perche piu può quale quando quello questa queste questi questo
        sono sua sue sui sul sulla suo tra tutti tutto una uno vari verso voi""".split()),
    "nld": set("""aan als bij dan dat deze die dit door een en het hij hun ist kan kunnen maar
        meer met naar niet nog om ons onze ook op over te tot uit van voor waar wat werd wij worden
        wordt zal zich zij zijn een andere alle bent geen heb hebben heeft hier iets naast onder
        veel volgens waarbij welke zeer zonder""".split()),
    "rus": set("""без более быть был была были было вам вас весь все всех вы да для до его ее если
        есть еще же за из или им их как ко когда которые кто ли мы на над нас не него нее нет них
        но об они она оно от по под при про так также те тем то тоже только том тот ты у уже чем
        что чтобы эта эти это этой этом этот""".split()),
    "tur": set("""ama ancak bir birlikte bu bunlar bunu çok da daha de değil diğer eğer en gibi
        hem her için ile ise kadar ki mi mu mı ne niye o olan olarak olduğu olur sonra şey şu tüm
        va ve veya ya yani""".split()),
    "pol": set("""aby albo ale bez być czy dla do gdy gdzie ich ile inne jak jako jest jego jej
        już które który lub ma mnie może na nad nie o od oraz po pod przez przy raz się są tak
        także tego tej ten to tym w we więc wszystkie z za ze że""".split()),
}


def _load_topic_corpus(derived: Path) -> pd.DataFrame:
    path = derived / "topic_corpus"
    return ds.dataset(str(path), format="parquet").to_table().to_pandas()


def _eligible(derived: Path, langs: list[str], template_min_cluster: int) -> pd.DataFrame:
    feats = load_features(
        derived,
        columns=["shard", "rg", "rg_row", "dataset_index", "url", "h1_text", "doc_kind", "lang",
                 "skeleton_sha1", "content_sha1", "generator_source", "generator_id",
                 "n_chars", "n_links"],
    )
    feats = add_derived(feats, template_min_cluster)
    feats = feats[feats["is_human_authored"]]
    corpus = _load_topic_corpus(derived)
    key = ["shard", "rg", "rg_row"]
    df = corpus.merge(feats[[*key, "dataset_index", "url", "h1_text", "skeleton_sha1"]],
                      on=key, how="inner", suffixes=("", "_f"))
    df = df[df["lang"].isin(langs)]
    # One document per template skeleton, so repeated boilerplate cannot
    # dominate a topic.
    df = df.drop_duplicates(subset=["skeleton_sha1"])
    return df


def fit(derived: str | Path, languages: list[str], out: str | Path,
        n_topics: int = DEFAULT_N_TOPICS, min_docs: int = DEFAULT_MIN_DOCS,
        template_min_cluster: int = 50, max_docs: int = 200_000) -> dict:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    derived = Path(derived)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    langs = [l for l in languages if l not in UNSUPPORTED_SCRIPTS]
    skipped = sorted(set(languages) - set(langs))
    if skipped:
        print(f"skipping languages without whitespace segmentation: {', '.join(skipped)}")

    pool = _eligible(derived, langs, template_min_cluster)
    result: dict = {"languages": {}, "skipped_languages": skipped,
                    "n_topics": n_topics, "min_docs": min_docs}

    for lang in langs:
        sub = pool[pool["lang"] == lang]
        if len(sub) < min_docs:
            print(f"  {lang}: {len(sub)} docs < min_docs={min_docs}, skipped")
            continue
        if len(sub) > max_docs:
            sub = sub.sample(max_docs, random_state=0)
        texts = sub["text"].tolist()
        print(f"  {lang}: fitting LDA on {len(texts)} documents ...")

        # A term must appear in at least min_df documents and at most 35% of
        # them. Clamp min_df below the max_df cut-off so that small corpora
        # (tests, rare languages) cannot produce an empty vocabulary.
        max_df = 0.35
        min_df = max(2, min(max(10, len(texts) // 2000), int(max_df * len(texts))))
        stop = "english" if lang == "eng" else (sorted(STOPWORDS[lang]) if lang in STOPWORDS else None)
        vec = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
            stop_words=stop,
            min_df=min_df,
            max_df=max_df,
            max_features=20000,
        )
        X = vec.fit_transform(texts)
        lda = LatentDirichletAllocation(
            n_components=n_topics, learning_method="online", batch_size=4096,
            max_iter=8, random_state=0, n_jobs=1,
        )
        doc_topic = lda.fit_transform(X)
        assign = doc_topic.argmax(axis=1)
        vocab = vec.get_feature_names_out()

        topics = []
        for k in range(n_topics):
            terms = [vocab[i] for i in lda.components_[k].argsort()[: -TOP_TERMS - 1 : -1]]
            idx = [i for i in (-doc_topic[:, k]).argsort()[: N_EXAMPLES * 4]
                   if assign[i] == k][:N_EXAMPLES]
            examples = [
                {"h1": str(sub.iloc[i]["h1_text"])[:90], "url": str(sub.iloc[i]["url"]),
                 "idx": int(sub.iloc[i]["dataset_index"])}
                for i in idx
            ]
            n_k = int((assign == k).sum())
            topics.append({
                "id": k, "terms": terms, "n_docs": n_k,
                "share": round(100.0 * n_k / len(texts), 2), "examples": examples,
            })
        topics.sort(key=lambda t: -t["n_docs"])
        result["languages"][lang] = {"n_docs": len(texts), "topics": topics}

        txt = out / f"topics_{lang}.txt"
        with txt.open("w") as fh:
            fh.write(f"# LDA topics for lang={lang}  ({len(texts)} documents, {n_topics} topics)\n")
            fh.write("# Copy the ids into topic_names.yaml and give each a human-readable name.\n\n")
            for t in topics:
                fh.write(f"[{lang}:{t['id']}]  {t['share']}%  ({t['n_docs']} docs)\n")
                fh.write("  terms: " + ", ".join(t["terms"]) + "\n")
                for ex in t["examples"]:
                    fh.write(f"    - [{ex['idx']}] {ex['h1']}  <{ex['url']}>\n")
                fh.write("\n")
        print(f"  {lang}: wrote {txt}")

    (out / "topics_raw.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {out/'topics_raw.json'}")
    return result


_YAML_LINE = re.compile(r"^\s*([\"']?)([\w:.-]+)\1\s*:\s*(.+?)\s*$")


def load_names(path: str | Path) -> dict[str, str]:
    """Parse ``topic_names.yaml``: flat ``<lang>:<id>: Human name`` mapping."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(p.read_text()) or {}
        return {str(k): str(v) for k, v in data.items()}
    except ImportError:
        names = {}
        for line in p.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _YAML_LINE.match(line)
            if m:
                names[m.group(2)] = m.group(3).strip("\"'")
        return names


def assign(raw_path: str | Path, names_path: str | Path, out: str | Path) -> dict:
    raw = json.loads(Path(raw_path).read_text())
    names = load_names(names_path)
    missing = []
    for lang, block in raw["languages"].items():
        for t in block["topics"]:
            key = f"{lang}:{t['id']}"
            t["name"] = names.get(key) or f"unlabeled-{t['id']}"
            if key not in names:
                missing.append(key)
    raw["unlabeled"] = missing
    Path(out).write_text(json.dumps(raw, indent=1))
    if missing:
        print(f"warning: {len(missing)} unlabeled topics: {', '.join(missing[:10])}")
    print(f"wrote {out}")
    return raw
