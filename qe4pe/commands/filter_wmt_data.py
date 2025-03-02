"""Select data from the WMT subset to match QE4PE requirements."""

import json
import logging
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import stanza
from tqdm import tqdm
from typer import Argument, Option

logger = logging.getLogger(__name__)

TGT_LANGS = ["ita", "nld"]
COMET_SIZES = ["xl", "xxl"]
NLLB_SIZES = ["600m", "3b"]
ERROR_TYPES = ["minor", "major", "critical"]

TASKS = ["pretask", "main", "posttask"]


def get_errors(entry, error_type=None):
    if (isinstance(entry, float) and math.isnan(entry)) or entry is None:
        return 0
    elif error_type is None:
        return len(entry)
    else:
        return len([e for e in entry if e["severity"] == error_type])


def get_words_in_errors(entry, error_type=None):
    if (isinstance(entry, float) and math.isnan(entry)) or entry is None:
        return 0
    elif error_type is None:
        return sum(len(err["text"].split()) for err in entry)
    else:
        return sum(len(err["text"].split()) for err in entry if err["severity"] == error_type)


def load_comet_json(path: str) -> list:
    with open(path) as f:
        jsonfile = json.load(f)
        out = []
        for comet_dic in jsonfile.values():
            out.extend(comet_dic)
    return out


def get_doc_df(df: pd.DataFrame, max_num_sentences: int = None):
    doc_entries = []
    for doc_id, group in df.groupby("doc_id"):
        num_segments = len(group)
        num_sentences = sum(group["num_sentences"])
        if max_num_sentences is not None and num_sentences >= max_num_sentences:
            cumsum = group["num_sentences"].cumsum()
            gt_max = cumsum >= max_num_sentences
            num_segments = list(gt_max).index(True) - 1
            if num_segments <= 0:
                continue
            group = group.iloc[:num_segments]
            num_sentences = list(cumsum)[num_segments]
        num_words = sum(group["num_words"])
        curr = {
            "position_id": group.iloc[0]["position_id"],
            "collection_id": group.iloc[0]["collection_id"],
            "doc_id": doc_id,
            "num_segments": num_segments,
            "num_sentences": num_sentences,
            "num_words": num_words,
        }
        for lang in TGT_LANGS:
            for nllb_size in NLLB_SIZES:
                curr[f"num_sentences_{lang}_{nllb_size}"] = sum(group[f"num_sentences_{lang}_{nllb_size}"])
                curr[f"num_words_{lang}_{nllb_size}"] = sum(group[f"num_words_{lang}_{nllb_size}"])
                for xcomet_size in COMET_SIZES:
                    idx = f"{xcomet_size}_{lang}_{nllb_size}"
                    curr[f"min_comet_{idx}"] = min(group[f"comet_{idx}"])
                    curr[f"avg_comet_{idx}"] = round(sum(group[f"comet_{idx}"]) / num_sentences, 4)
                    curr[f"num_all_errors_{idx}"] = sum(group[f"num_all_errors_{idx}"])
                    curr[f"avg_all_errors_{idx}"] = round(curr[f"num_all_errors_{idx}"] / num_sentences, 4)
                    curr[f"num_words_all_errors_{idx}"] = sum(group[f"num_words_all_errors_{idx}"])
                    curr[f"avg_words_all_errors_{idx}"] = round(curr[f"num_words_all_errors_{idx}"] / num_sentences, 4)
                    for error_type in ERROR_TYPES:
                        curr[f"num_{error_type}_errors_{idx}"] = sum(group[f"num_{error_type}_errors_{idx}"])
                        curr[f"avg_{error_type}_errors_{idx}"] = round(
                            curr[f"num_{error_type}_errors_{idx}"] / num_sentences, 4
                        )
                        curr[f"num_words_{error_type}_errors_{idx}"] = sum(
                            group[f"num_words_{error_type}_errors_{idx}"]
                        )
                        curr[f"avg_words_{error_type}_errors_{idx}"] = round(
                            curr[f"num_words_{error_type}_errors_{idx}"] / num_sentences, 4
                        )
        doc_entries.append(curr)
    doc_df = pd.DataFrame(doc_entries).sort_values("position_id")
    return doc_df


def get_idx(df, doc_id, seg_id):
    return df[(df["doc_id"] == f"doc{doc_id}") & (df["seg_id"] == seg_id)].index.values[0]


def filter_wmt_data(
    wmt23_path: Annotated[
        str, Argument(..., help="Path to the WMT23 data that require filtering. Default: data/setup/wmt23")
    ] = None,
    output_csv_path: Annotated[
        str,
        Option(..., help="Path to the folder containing intermediate output files. Defaults to data/setup/processed"),
    ] = None,
    use_saved_data: Annotated[
        bool, Option(..., help="Use saved intermediate data to avoid reprocessing. Default: False")
    ] = False,
):
    """Filter the WMT data to match QE4PE requirements."""
    if wmt23_path is None:
        wmt23_path = Path("data/setup/wmt23")
    if output_csv_path is None:
        output_csv_path = Path("data/setup/processed")

    if use_saved_data and (output_csv_path / "orig_segment_df.csv").exists():
        logger.info("Loading saved data 'orig_segment_df.csv' from %s", output_csv_path)
        logger.info("Set 'use_saved_data' to False to reprocess the data instead.")
        base_df = pd.read_csv(output_csv_path / "orig_segment_df.csv")
        doc_df = pd.read_csv(output_csv_path / "orig_doc_df.csv")
    else:
        nlp_en = stanza.Pipeline("en", processors="tokenize")
        nlp_it = stanza.Pipeline("it", processors="tokenize")
        nlp_nl = stanza.Pipeline("nl", processors="tokenize")

        def get_nlp(lang):
            if lang == "eng":
                return nlp_en
            elif lang == "ita":
                return nlp_it
            elif lang == "nld":
                return nlp_nl
            else:
                return None

        def count_sentences(text, lang):
            doc = get_nlp(lang)(text)
            return len(doc.sentences)

        # Step 1: Load the data
        logger.info("Loading data from %s", wmt23_path)
        dic_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        with open(wmt23_path / "wmttest2023.eng") as f:
            dic_files["eng"]["texts"] = [l.strip() for l in f.readlines()]

        with open(wmt23_path / "wmttest2023.eng.jsonl") as f:
            eng_meta = []
            for line in f:
                eng_meta.append(json.loads(line))

        for lang, nllb_size in list(product(TGT_LANGS, NLLB_SIZES)):
            with open(wmt23_path / f"nllb_{nllb_size}" / f"wmttest2023.{lang}") as f:
                dic_files[lang][nllb_size]["texts"] = [l.strip() for l in f.readlines()]
            for size in COMET_SIZES:
                dic_files[lang][nllb_size][f"scores_{size}"] = load_comet_json(
                    wmt23_path / f"nllb_{nllb_size}" / f"wmttest2023_xcomet-{size}_{lang}.json"
                )

        logger.info("=" * 10 + " Total number of segments " + "=" * 10)
        logger.info("WMT23 eng: %i", len(dic_files["eng"]["texts"]))
        logger.info("WMT23 eng meta: %i", len(eng_meta))

        for lang, nllb_size in list(product(TGT_LANGS, NLLB_SIZES)):
            logger.info(f"WMT23 {lang} {nllb_size}", len(dic_files[lang][nllb_size]["texts"]))
            for comet_size in COMET_SIZES:
                score_name = f"scores_{comet_size}"
                logger.info(f"WMT23 {lang} {nllb_size} {comet_size}: {len(dic_files[lang][nllb_size][score_name])}")

        # Step 2: Build an initial dataframe
        data_info = []
        for i in range(len(eng_meta)):
            curr = {
                "collection_id": eng_meta[i]["collection_id"],
                "doc_id": eng_meta[i]["doc_id"],
                "seg_id": eng_meta[i]["seg_id"],
                "eng": eng_meta[i]["src"],
            }
            for lang, nllb_size in list(product(TGT_LANGS, NLLB_SIZES)):
                curr[f"{lang}_{nllb_size}"] = dic_files[lang][nllb_size]["texts"][i]
                for comet_size in COMET_SIZES:
                    curr[f"comet_{comet_size}_{lang}_{nllb_size}"] = round(
                        dic_files[lang][nllb_size][f"scores_{comet_size}"][i]["COMET"], 4
                    )
                    curr[f"errors_{comet_size}_{lang}_{nllb_size}"] = dic_files[lang][nllb_size][
                        f"scores_{comet_size}"
                    ][i].get("errors", None)
            data_info.append(curr)

        base_df = pd.DataFrame(data_info)

        # Step 3: Create extra fields for word/sentences/error spans to inform selection
        base_df.insert(0, "position_id", range(1, 1 + len(base_df)))
        base_df.insert(4, "num_sentences", [count_sentences(x, "eng") for x in tqdm(base_df["eng"])])
        base_df.insert(5, "num_words", [len(x.split()) for x in base_df["eng"]])

        for lang, nllb_size in tqdm(list(product(TGT_LANGS, NLLB_SIZES))):
            base_df[f"num_sentences_{lang}_{nllb_size}"] = base_df[f"{lang}_{nllb_size}"].apply(
                lambda entry: count_sentences(entry, lang)
            )
            base_df[f"num_words_{lang}_{nllb_size}"] = base_df[f"{lang}_{nllb_size}"].apply(
                lambda entry: len(entry.split()) if isinstance(entry, str) else 0
            )
            for xcomet_size in COMET_SIZES:
                idx = f"{xcomet_size}_{lang}_{nllb_size}"
                base_df[f"num_all_errors_{idx}"] = base_df[f"errors_{idx}"].apply(lambda entry: get_errors(entry))
                base_df[f"num_words_all_errors_{idx}"] = base_df[f"errors_{idx}"].apply(
                    lambda entry: get_words_in_errors(entry)
                )
                for error_type in ERROR_TYPES:
                    base_df[f"num_{error_type}_errors_{idx}"] = base_df[f"errors_{idx}"].apply(
                        lambda entry: get_errors(entry, error_type)
                    )
                    base_df[f"num_words_{error_type}_errors_{idx}"] = base_df[f"errors_{idx}"].apply(
                        lambda entry: get_words_in_errors(entry, error_type)
                    )
        base_df.to_csv(output_csv_path / "orig_segment_df.csv", index=False)

        # Step 4: Build doc-level dataframe
        doc_df = get_doc_df(base_df)
        doc_df.to_csv(output_csv_path / "orig_doc_df.csv", index=False)
    logger.info("Original segment dataframe shape: %s", base_df.shape)
    logger.info("Doc dataframe shape: %s", doc_df.shape)

    # Step 5: Filter the data according to the QE4PE requirements
    doc_df_filtered = get_doc_df(base_df, max_num_sentences=10)
    doc_df_filtered = doc_df_filtered[
        # At least 4 sentences per document
        (doc_df_filtered["num_sentences"] >= 4)
        & (doc_df_filtered["num_sentences_ita_3b"] >= 4)
        & (doc_df_filtered["num_sentences_nld_3b"] >= 4)
        &
        ## Average sentence-level COMET is between 0.3 and 0.95, with a minimum of 0.3 per segment
        (doc_df_filtered["avg_comet_xxl_ita_3b"] >= 0.3)
        & (doc_df_filtered["avg_comet_xxl_nld_3b"] >= 0.3)
        & (doc_df_filtered["avg_comet_xxl_ita_3b"] < 0.95)
        & (doc_df_filtered["avg_comet_xxl_nld_3b"] < 0.95)
        & (doc_df_filtered["min_comet_xxl_ita_3b"] >= 0.3)
        & (doc_df_filtered["min_comet_xxl_nld_3b"] >= 0.3)
        &
        ## At least 3 and at most 20 errors
        (doc_df_filtered["num_all_errors_xxl_ita_3b"] >= 3)
        & (doc_df_filtered["num_all_errors_xxl_ita_3b"] <= 20)
        & (doc_df_filtered["num_all_errors_xxl_nld_3b"] >= 3)
        & (doc_df_filtered["num_all_errors_xxl_nld_3b"] <= 20)
        &
        # At least 10 words per sentence, and at most 100
        (doc_df_filtered["num_words"] >= (5 * doc_df_filtered["num_sentences"]))
        & (doc_df_filtered["num_words"] <= (100 * doc_df_filtered["num_sentences"]))
        & (doc_df_filtered["num_words_ita_3b"] >= (5 * doc_df_filtered["num_sentences_ita_3b"]))
        & (doc_df_filtered["num_words_ita_3b"] <= (100 * doc_df_filtered["num_sentences_ita_3b"]))
        & (doc_df_filtered["num_words_nld_3b"] >= (5 * doc_df_filtered["num_sentences_nld_3b"]))
        & (doc_df_filtered["num_words_nld_3b"] <= (100 * doc_df_filtered["num_sentences_nld_3b"]))
        &
        ## No more than 30% of text is highlighted
        (doc_df_filtered["num_words_all_errors_xxl_ita_3b"] <= (0.3 * doc_df_filtered["num_words_ita_3b"]))
        & (doc_df_filtered["num_words_all_errors_xxl_nld_3b"] <= (0.3 * doc_df_filtered["num_words_nld_3b"]))  # &
    ]
    logger.info("Post-filter doc dataframe shape: %s", doc_df_filtered.shape)
    doc_df_filtered.to_csv(output_csv_path / "filtered_doc_df.csv", index=False)
    doc_ids = list(doc_df_filtered.doc_id)
    num_segments_per_doc = list(doc_df_filtered.num_segments)

    # LEGEND of collection_id:
    # user.XXXXXXXXXXX:
    #   Mastodon posts
    # rs-XXXX, nm-XXXX, rn-XXXX, rs-XXXX:
    #   RoCS-MT: Robustness Challenge Set for Machine Translation (https://aclanthology.org/2023.wmt-1.21/)
    #   NOTE: Same ID number=same source document here!
    # docXX:
    #   WMT23 Biomedical Translation Task (https://aclanthology.org/2023.wmt-1.2/)
    with open(output_csv_path / "filtered_doc_ids.txt", "w") as f:
        f.write("\n".join(doc_ids))
    with open(output_csv_path / "filtered_doc_num_segments_per_doc.txt", "w") as f:
        f.write("\n".join([str(n) for n in num_segments_per_doc]))
    num_segments_per_doc = dict(zip(doc_ids, num_segments_per_doc, strict=False))

    # Step 6: Apply doc-level filtering to segment-level dataframe
    base_df_filtered = base_df[base_df["doc_id"].isin(doc_ids)]

    for doc_id, num_segments in num_segments_per_doc.items():
        base_df_filtered = base_df_filtered[
            ~((base_df_filtered["doc_id"] == doc_id) & (base_df_filtered["seg_id"] > num_segments))
        ]
    logger.info("Post-filter segment dataframe shape: %s", base_df_filtered.shape)
    base_df_filtered.to_csv(output_csv_path / "filtered_segment_df.csv", index=False)

    # Step 7: Create input files for the QE4PE task
    for task in TASKS:
        task_config = json.load(open(Path("data") / "task" / task / "doc_id_map.json"))
        task_ids_map = {v: k for k, v in task_config["map"].items()}
        logger.info("Creating input files for %s (%s docs)", task, len(task_ids_map.values()))
        task_df_filtered = base_df_filtered[base_df_filtered["doc_id"].isin(task_ids_map.keys())]
        logger.info("ENG - %s segments, %s words", task_df_filtered.shape[0], task_df_filtered["num_words"].sum())
        logger.info(
            "ITA - %s segments, %s words", task_df_filtered.shape[0], task_df_filtered["num_words_ita_3b"].sum()
        )
        logger.info(
            "NLD - %s segments, %s words", task_df_filtered.shape[0], task_df_filtered["num_words_nld_3b"].sum()
        )
        task_df_filtered["doc_id"] = task_df_filtered["doc_id"].apply(lambda x: task_ids_map[x])
        task_df_filtered.to_csv(output_csv_path / f"{task}_segment_df.csv", index=False)

    # Step 8: Add manual critical errors for the main task
    main_df = pd.read_csv(output_csv_path / "main_segment_df.csv")
    np.random.seed(42)
    doc_ids = main_df["doc_id"].unique()
    np.random.shuffle(doc_ids)
    shuffled_df = pd.concat([main_df[main_df["doc_id"] == doc_id] for doc_id in doc_ids])

    # Manually editing 15 instances to introduce critical errors:

    # 1-8: Change 12 to 11
    idx = get_idx(shuffled_df, 1, 8)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("12", "11")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("12", "11")

    # 13-5: Change clearly to mildly (duidelijk -> licht)
    idx = get_idx(shuffled_df, 13, 5)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("duidelijk", "licht")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("chiaramente", "lievemente")

    # 13-6: Change "No significant change" to "Significant change" (e.g. Er werden geen significante verschillen gevonden -> Er werden significante verschillen gevonden)
    idx = get_idx(shuffled_df, 13, 6)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace(
        "Er werden geen significante verschillen gevonden", "Er werden significante verschillen gevonden"
    )
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace(
        "Non sono state riscontrate differenze significative", "Sono state riscontrate differenze significative"
    )

    # 16-3: Change "The Last of Us" with literal translation of "the last of us" (e.g. het laatste van ons, l'ultimo di noi)
    idx = get_idx(shuffled_df, 16, 3)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("The Last of Us", "Het laatste van ons")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("The Last of Us", "L'ultimo di noi")

    # 18-3: Omit "Kaufman-Assessment Battery for Children" in the translation.
    idx = get_idx(shuffled_df, 18, 3)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace(
        ", 5-jarige Mental Processing Composite (MPC, Kaufman-Assessment Battery for Children) en Language Screening voor kleuters.",
        "en 5-jarige Mental Processing Composite (MPC, Language Screening for Preschoolers).",
    )
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace(
        "composto di elaborazione mentale a 5 anni (MPC, Kaufman-Assessment Battery for Children) e screening linguistico per bambini in età prescolare.",
        "composto di elaborazione mentale a 5 anni (MPC, screening linguistico per bambini in età prescolare).",
    )

    # 20-1: Change EMPoCUS to EmPoCU
    idx = get_idx(shuffled_df, 20, 1)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("EMPoCUS", "EmPoCU")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("EMPoCUS", "EmPoCU")

    # 20-7: Replace EMPoCUS with EmPoCU and PoCUS with PoCU
    idx = get_idx(shuffled_df, 20, 7)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("EMPoCUS", "EmPoCU").replace("PoCUS", "PoCU")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("EMPoCUS", "EmPoCU").replace("PoCUS", "PoCU")

    # 22-1: Change "#Mastodon" with #Mastodont
    idx = get_idx(shuffled_df, 22, 1)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("#Mastodon", "#Mastodont")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("#Mastodon", "#Mastodonte")

    # 23-4: Add "but I think I can do it." at the end of the translation.
    idx = get_idx(shuffled_df, 23, 4)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace(
        "2 maanden dichtbij studeer.", "2 maanden dichtbij studeer, maar ik denk dat ik het kan."
    )
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace(
        "a studiare per 2 mesi.", "a studiare per 2 mesi, ma penso di potercela fare."
    )

    # 31-2: Change CARE to CRE
    idx = get_idx(shuffled_df, 31, 2)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("CARE", "CRE")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("CARE", "CRE")

    # 34-7: Change 29.20±8.06 to 29.06±8.20
    idx = get_idx(shuffled_df, 34, 7)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("29,20 ± 8,06", "29,06 ± 8,20")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("29,20±8,06", "29,06±8,20")

    # 37-4: Replace decrease with increase (e.g. verminderen -> verhogen)
    idx = get_idx(shuffled_df, 37, 4)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("verminderen", "verhogen")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("ridurre", "incrementare")

    # 43-5: Change Kara to Carolin
    idx = get_idx(shuffled_df, 43, 5)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace("Kara", "Carolin")
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace("Kara", "Carolina")

    # 48-5: Change "alkaline phosphatase" to the translation of "protein kinase"
    idx = get_idx(shuffled_df, 48, 5)
    nld_str = shuffled_df.at[idx, "nld_3b"]
    ita_str = shuffled_df.at[idx, "ita_3b"]
    shuffled_df.at[idx, "nld_3b"] = nld_str.replace(
        "met uitzondering van alkalische fosfatase.", "met uitzondering van protein kinase."
    )
    shuffled_df.at[idx, "ita_3b"] = ita_str.replace(
        "ad eccezione della fosfatasi alcalina.", "ad eccezione della chinasi proteica."
    )
    shuffled_df.to_csv(output_csv_path / "main_segment_df.csv", index=False)

    # Replace the column names with "Document ID", "Segment ID", "English Source", "Dutch Translation"
    # Replace document IDs with a sequential number
    # Write doc_id, seg_id, eng and nld_3b to an xslx file
    shuffled_df = shuffled_df.rename(
        columns={
            "doc_id": "Document ID",
            "seg_id": "Segment ID",
            "eng": "English Source",
            "nld_3b": "Dutch Translation",
            "ita_3b": "Italian Translation",
        }
    )
    shuffled_df.insert(3, "Document Index", 0)
    for idx, doc_id in enumerate(shuffled_df["Document ID"].unique(), start=1):
        shuffled_df.loc[shuffled_df["Document ID"] == doc_id, "Document Index"] = idx
    shuffled_df[["Document Index", "Segment ID", "English Source", "Dutch Translation"]].to_excel(
        output_csv_path / "nld_oracle_pe_assignment.xlsx", index=False
    )
    shuffled_df[["Document Index", "Segment ID", "English Source", "Italian Translation"]].to_excel(
        output_csv_path / "ita_oracle_pe_assignment.xlsx", index=False
    )


def filter_wmt_data_callback(verbose: Annotated[bool, Option(..., help="Increase verbosity")] = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
