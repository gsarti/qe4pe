import datetime
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Annotated, Any

import jiwer
import pandas as pd
from typer import Argument, Option

logger = logging.getLogger(__name__)


@dataclass
class DocBehavioralMetrics:
    doc_id: int = field(metadata={"description": "The index of the document in the current configuration of the QE4PE dataset containing the current segment."})
    num_edits: int = field(default=None, metadata={"description": "Total number of edits performed by the translator on the current document. Only the last edit outputs are considered valid."})
    edit_order: int = field(default=None, metadata={"description": "Index corresponding to the current document edit order. If equal to doc_id, the document was edited in the given order."})
    edit_time: float = field(default=None, metadata={"description": "Total editing time for the current document in seconds (from start to end, no times ignored)."})
    edit_time_filtered: float = field(default=None, metadata={"description": "Total editing time for the current document in seconds (from start to end, >5m pauses between logged actions ignored)."})
    keys_per_min: float = field(default=None, metadata={"description": "Keystrokes per minute computed for the current document using doc_edit_time_filtered over the number of change actions."})
    chars_per_min: float = field(default=None, metadata={"description": "Characters per minute computed for the current document using doc_edit_time_filtered over the machine-translated text."})
    words_per_min: float = field(default=None, metadata={"description": "Words per minute computed for the current document using doc_edit_time_filtered over the machine-translated text."})


@dataclass
class QE4PEProcessedEntry:
    # Identification
    unit_id: str = field(default=None, metadata={"description": "The full entry identifier. Format: qe4pe-{task_id}-{src_lang}-{tgt_lang}-{doc_id}-{segment_in_doc_id}-{translator_main_task_id}."})
    wmt_id: str = field(default=None, metadata={"description": "Identifier of the sentence in the original WMT23 dataset."})
    wmt_category: str = field(default=None, metadata={"description": "Category of the document: `biomedical` or `social`."})
    doc_id: int = field(default=None, metadata={"description": "The index of the document in the current configuration of the QE4PE dataset containing the current segment."})
    segment_in_doc_id: int = field(default=None, metadata={"description": "The index of the segment inside the current document."})
    segment_id: int = field(default=None, metadata={"description": "The index of the segment in the current configurations (i.e. concatenating all segments from all documents in order)."})
    translator_pretask_id: str = field(default=None, metadata={"description": "The identifier for the translator according to the pretask format before modality assignments: tXX."})
    translator_main_id: str = field(default=None, metadata={"description": "The identifier for the translator according to the main task format after modality assignments: {highlight_modality}_tXX."})
    src_lang: str = field(default=None, metadata={"description": "The source language of the segment. For QE4PE, this is always English (eng)."})
    tgt_lang: str = field(default=None, metadata={"description": "The target language of the segment: either Italian (ita) or Dutch (nld)."})
    highlight_modality: str = field(default=None, metadata={"description": "The highlighting modality used for the segment. Values: no_highlight, oracle, supervised, unsupervised."})
    has_issue: bool = field(default=False, metadata={"description": "If True, the segment has an issue and should be excluded from the analysis."})
    issue_description: str = field(default=None, metadata={"description": "Description of the issue detected in the segment. Empty if has_issue is False."})
    has_added_critical_error: bool = field(default=False, metadata={"description": "If True, a critical error was manually added to the MT output."})
    critical_error_description: str = field(default=None, metadata={"description": "Description of the critical error added to the MT output. Empty if has_added_critical_error is False."})

    # Text statistics
    src_num_chars: int = field(default=None, metadata={"description": "Length of the source segment in number of characters."})
    mt_num_chars: int = field(default=None, metadata={"description": "Length of the machine-translated segment in number of characters."})
    pe_num_chars: int = field(default=None, metadata={"description": "Length of the post-edited segment in number of characters."})
    src_num_words: int = field(default=None, metadata={"description": "Length of the source segment in number of words."})
    mt_num_words: int = field(default=None, metadata={"description": "Length of the machine-translated segment in number of words."})
    pe_num_words: int = field(default=None, metadata={"description": "Length of the post-edited segment in number of words."})
    num_minor_highlighted_chars: int = field(default=None, metadata={"description": "Number of characters highlighted as minor errors in the machine-translated text."})
    num_major_highlighted_chars: int = field(default=None, metadata={"description": "Number of characters highlighted as major errors in the machine-translated text."})
    num_minor_highlighted_words: int = field(default=None, metadata={"description": "Number of words highlighted as minor errors in the machine-translated text."})
    num_major_highlighted_words: int = field(default=None, metadata={"description": "Number of words highlighted as major errors in the machine-translated text."})

    # Edits statistics
    num_words_insert: int = field(default=None, metadata={"description": "Number of post-editing insertions computed using jiwer."})
    num_words_delete: int = field(default=None, metadata={"description": "Number of post-editing deletions computed using jiwer."})
    num_words_substitute: int = field(default=None, metadata={"description": "Number of post-editing substitutions computed using jiwer."})
    num_words_unchanged: int = field(default=None, metadata={"description": "Number of post-editing hits computed using jiwer."})
    tot_words_edits: int = field(default=None, metadata={"description": "Total of all edit types for the sentence."})
    wer: float = field(default=None, metadata={"description": "Word Error Rate score computed between mt_text and pe_text using jiwer."})
    num_chars_insert: int = field(default=None, metadata={"description": "Number of post-editing character insertions computed using jiwer."})
    num_chars_delete: int = field(default=None, metadata={"description": "Number of post-editing character deletions computed using jiwer."})
    num_chars_substitute: int = field(default=None, metadata={"description": "Number of post-editing character substitutions computed using jiwer."})
    num_chars_unchanged: int = field(default=None, metadata={"description": "Number of post-editing character hits computed using jiwer."})
    tot_chars_edits: int = field(default=None, metadata={"description": "Total of all character edit types for the sentence."})
    cer: float = field(default=None, metadata={"description": "Character Error Rate score computed between mt_text and pe_text using jiwer."})

    # Translation quality
    mt_bleu_max: float = field(default=None, metadata={"description": "Max BLEU score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_bleu_min: float = field(default=None, metadata={"description": "Min BLEU score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_bleu_mean: float = field(default=None, metadata={"description": "Mean BLEU score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_bleu_std: float = field(default=None, metadata={"description": "Standard deviation of BLEU scores between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_chrf_max: float = field(default=None, metadata={"description": "Max chrF score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_chrf_min: float = field(default=None, metadata={"description": "Min chrF score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_chrf_mean: float = field(default=None, metadata={"description": "Mean chrF score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_chrf_std: float = field(default=None, metadata={"description": "Standard deviation of chrF scores between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_ter_max: float = field(default=None, metadata={"description": "Max TER score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_ter_min: float = field(default=None, metadata={"description": "Min TER score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_ter_mean: float = field(default=None, metadata={"description": "Mean TER score between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_ter_std: float = field(default=None, metadata={"description": "Standard deviation of TER scores between mt_text and all pe_text for the corresponding segment using SacreBLEU with default parameters."})
    mt_comet_max: float = field(default=None, metadata={"description": "Max COMET sentence-level score for the mt_text and all pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    mt_comet_min: float = field(default=None, metadata={"description": "Min COMET sentence-level score for the mt_text and all pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    mt_comet_mean: float = field(default=None, metadata={"description": "Mean COMET sentence-level score for the mt_text and all pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    mt_comet_std: float = field(default=None, metadata={"description": "Standard deviation of COMET sentence-level scores for the mt_text and all pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    mt_xcomet_qe: float = field(default=None, metadata={"description": "XCOMET-XXL sentence-level quality estimation score for the mt_text."})
    mt_xcomet_errors: list[dict[str, str | int | float]] = field(default=None, metadata={"description": "List of error spans detected by XCOMET-XXL for the mt_text."})
    pe_bleu_max: float = field(default=None, metadata={"description": "Max BLEU score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_bleu_min: float = field(default=None, metadata={"description": "Min BLEU score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_bleu_mean: float = field(default=None, metadata={"description": "Mean BLEU score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_bleu_std: float = field(default=None, metadata={"description": "Standard deviation of BLEU scores between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_chrf_max: float = field(default=None, metadata={"description": "Max chrF score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_chrf_min: float = field(default=None, metadata={"description": "Min chrF score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_chrf_mean: float = field(default=None, metadata={"description": "Mean chrF score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_chrf_std: float = field(default=None, metadata={"description": "Standard deviation of chrF scores between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_ter_max: float = field(default=None, metadata={"description": "Max TER score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_ter_min: float = field(default=None, metadata={"description": "Min TER score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_ter_mean: float = field(default=None, metadata={"description": "Mean TER score between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_ter_std: float = field(default=None, metadata={"description": "Standard deviation of TER scores between pe_text and all other pe_text for the corresponding segment using SacreBLEU with default parameters."})
    pe_comet_max: float = field(default=None, metadata={"description": "Max COMET sentence-level score for the pe_text and all other pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    pe_comet_min: float = field(default=None, metadata={"description": "Min COMET sentence-level score for the pe_text and all other pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    pe_comet_mean: float = field(default=None, metadata={"description": "Mean COMET sentence-level score for the pe_text and all other pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    pe_comet_std: float = field(default=None, metadata={"description": "Standard deviation of COMET sentence-level scores for the pe_text and all other pe_text for the corresponding segment using Unbabel/wmt22-comet-da with default parameters."})
    pe_xcomet_qe: float = field(default=None, metadata={"description": "XCOMET-XXL sentence-level quality estimation score for the pe_text."})
    pe_xcomet_errors: list[dict[str, str | int | float]] = field(default=None, metadata={"description": "List of error spans detected by XCOMET-XXL for the pe_text."})

    # Behavioral data
    doc_num_edits: int = field(default=None, metadata={"description": "Total number of edits performed by the translator on the current document. Only the last edit outputs are considered valid."})
    doc_edit_order: int = field(default=None, metadata={"description": "Index corresponding to the current document edit order. If equal to doc_id, the document was edited in the given order."})
    doc_edit_time: float = field(default=None, metadata={"description": "Total editing time for the current document in seconds (from start to end, no times ignored)."})
    doc_edit_time_filtered: float = field(default=None, metadata={"description": "Total editing time for the current document in seconds (from start to end, >5m pauses between logged actions ignored)."})
    doc_keys_per_min: float = field(default=None, metadata={"description": "Keystrokes per minute computed for the current document using doc_edit_time_filtered over the number of change actions."})
    doc_chars_per_min: float = field(default=None, metadata={"description": "Characters per minute computed for the current document using doc_edit_time_filtered over the machine-translated text."})
    doc_words_per_min: float = field(default=None, metadata={"description": "Words per minute computed for the current document using doc_edit_time_filtered over the machine-translated text."})
    segment_num_edits: int = field(default=None, metadata={"description": "Total number of edits performed by the translator on the current segment. Only the edits in the last doc edit are considered valid."})
    segment_edit_order: int = field(default=None, metadata={"description": "Index corresponding to the current segment edit order (only first enter action counts). If equal to segment_in_doc_id, the segment was edited in the given order."})
    segment_edit_time: float = field(default=None, metadata={"description": "Total editing time for the current segment in seconds (summed time between enter-exit blocks)."})
    segment_edit_time_filtered: float = field(default=None, metadata={"description": "Total editing time for the current segment in seconds (>5m pauses between logged actions ignored)."})
    segment_keys_per_min: float = field(default=None, metadata={"description": "Keystrokes per minute computed for the current document using doc_edit_time_filtered over the number of change actions."})
    segment_chars_per_min: float = field(default=None, metadata={"description": "Characters per minute computed for the current document using doc_edit_time_filtered over the machine-translated text."})
    segment_words_per_min: float = field(default=None, metadata={"description": "Words per minute computed for the current document using doc_edit_time_filtered over the machine-translated text."})
    num_enter_actions: int = field(default=None, metadata={"description": "Number of enter actions (focus on textbox) performed by the translator on the current segment during post-editing."})
    remove_highlights: bool = field(default=None, metadata={"description": "If True, the Clear Highlights button was pressed for this segment (always False for no_highlight modality)."})

    # Texts and annotations
    src_text: str = field(default=None, metadata={"description": "The original source segment from WMT23 requiring translation."})
    mt_text: str = field(default=None, metadata={"description": "Output of the NLLB-3.3B model when translating src_text into tgt_lang (default config, 5 beams)."})
    mt_text_highlighted: str = field(default=None, metadata={"description": "Highlighted version of mt_text with potential errors according to the highlight_modality."})
    pe_text: str = field(default=None, metadata={"description": "Post-edited version of mt_text produced by a professional translator with highlight_modality."})
    mt_pe_word_aligned: str = field(default=None, metadata={"description": "Aligned visual representation of word-level edit operations (I = Insertion, D = Deletion, S = Substitution)."})
    mt_pe_char_aligned: str = field(default=None, metadata={"description": "Aligned visual representation of character-level edit operations (I = Insertion, D = Deletion, S = Substitution)."})
    highlights: list[dict[str, str | int | float]] = field(default=None, metadata={"description": "List of dictionaries for highlighted spans with error severity and position, matching XCOMET format for word-level error annotations."})


def clean_tags(txt: str) -> str:
    return re.sub(r'<\/?(?:major|minor)>', '', txt)


def visualize_alignment(out: jiwer.WordOutput | jiwer.CharacterOutput) -> str:
    if out.insertions == 0 and out.deletions == 0 and out.substitutions == 0:
        join_char = " " if isinstance(out, jiwer.WordOutput) else ""
        lines = [f"MT: {join_char.join(out.references[0])}", f"PE: {join_char.join(out.hypotheses[0])}", ""]
    else:
        lines = jiwer.visualize_alignment(out).split("\n\n")[0].split("\n")[1:4]
        lines[0] = lines[0].replace("REF:", "MT:")
        lines[1] = lines[1].replace("HYP:", "PE:")
    return "\n".join(lines)

def get_highlighted_counts(text: str) -> dict[str, int]:
    """
    Analyze text with minor and major tags, returning a dictionary with character count and word count for each tag type.

    Args:
        text (str): Input text with <minor> and <major> tags
    """
    result = {'minor': {'chars': 0, 'words': 0}, 'major': {'chars': 0, 'words': 0}}
    minor_matches = re.findall(r'<minor>(.*?)</minor>', text)
    major_matches = re.findall(r'<major>(.*?)</major>', text)
    for match in minor_matches:
        result['minor']['chars'] += len(match)
        result['minor']['words'] += len(match.split())
    for match in major_matches:
        result['major']['chars'] += len(match)
        result['major']['words'] += len(match.split())
    return result


def add_identification_info(
    entry: QE4PEProcessedEntry,
    task_name: str,
    src_lang: str,
    tgt_lang: str,
    doc_id: int,
    segment_in_doc_id: int,
    segment_id: int,
    current_translator_id: str,
    pretask_translator_id: str,
    main_translator_id: str,
    modality: str,
    wmt_doc_map: dict[str, str],
    wmt_doc_category_map: dict[str, str],
    force_highlight_modality: str | None = None,
    entry_issue: dict[str, str | int] | None = None,
    entry_critical_error: dict[str, str | int] | None = None,
) -> QE4PEProcessedEntry:
    entry.unit_id = f"qe4pe-{task_name}-{src_lang}-{tgt_lang}-{doc_id}-{segment_in_doc_id}-{current_translator_id}"
    entry.wmt_id = wmt_doc_map[f"doc{doc_id}"]
    entry.wmt_category = wmt_doc_category_map[entry.wmt_id]
    entry.doc_id = doc_id
    entry.segment_in_doc_id = segment_in_doc_id
    entry.segment_id = segment_id
    entry.translator_pretask_id = pretask_translator_id
    entry.translator_main_id = main_translator_id
    entry.src_lang = src_lang
    entry.tgt_lang = tgt_lang
    entry.highlight_modality = modality if force_highlight_modality is None else force_highlight_modality
    if entry_issue is not None:
        entry.has_issue = True
        entry.issue_description = entry_issue["description"]
    else:
        entry.has_issue = False
        entry.issue_description = None
    if entry_critical_error is not None:
        entry.has_added_critical_error = True
        entry.critical_error_description = entry_critical_error["description"]
    else:
        entry.has_added_critical_error = False
        entry.critical_error_description = None
    return entry


def add_data_stats(
    entry: QE4PEProcessedEntry,
    src_text: str,
    mt_text: str,
    pe_text: str
) -> QE4PEProcessedEntry:
    entry.src_num_chars = len(src_text)
    entry.mt_num_chars = len(mt_text)
    entry.pe_num_chars = len(pe_text)
    entry.src_num_words = len(src_text.split())
    entry.mt_num_words = len(mt_text.split())
    entry.pe_num_words = len(pe_text.split())
    highlighted_counts = get_highlighted_counts(mt_text)
    entry.num_minor_highlighted_chars = highlighted_counts["minor"]["chars"]
    entry.num_major_highlighted_chars = highlighted_counts["major"]["chars"]
    entry.num_minor_highlighted_words = highlighted_counts["minor"]["words"]
    entry.num_major_highlighted_words = highlighted_counts["major"]["words"]
    return entry


def add_edits_stats(
    entry: QE4PEProcessedEntry,
    align_words: jiwer.WordOutput,
    align_chars: jiwer.CharacterOutput,
) -> QE4PEProcessedEntry:
    entry.num_words_insert = align_words.insertions
    entry.num_words_delete = align_words.deletions
    entry.num_words_substitute = align_words.substitutions
    entry.num_words_unchanged = align_words.hits
    entry.tot_words_edits = entry.num_words_insert + entry.num_words_delete + entry.num_words_substitute
    entry.wer = round(align_words.wer, 4)
    entry.num_chars_insert = align_chars.insertions
    entry.num_chars_delete = align_chars.deletions
    entry.num_chars_substitute = align_chars.substitutions
    entry.num_chars_unchanged = align_chars.hits
    entry.tot_chars_edits = entry.num_chars_insert + entry.num_chars_delete + entry.num_chars_substitute
    entry.cer = round(align_chars.cer, 4)
    return entry


def compute_time(df: pd.DataFrame, max_pause_threshold: datetime.timedelta | None = None) -> float:
    if len(df) == 0:
        return 0
    df = df.sort_values("time")
    start_time = df["time"].reset_index(drop=True)[0].timestamp()
    end_time = df["time"].reset_index(drop=True)[len(df["time"]) - 1].timestamp()
    timediff = end_time - start_time
    if max_pause_threshold is None:
        return int(timediff)
    times = df["time"].to_list()
    deltas = [b-a for a,b in zip(times, times[1:])]
    deltas = [x for x in deltas if x <= max_pause_threshold]
    return sum(deltas, start=datetime.timedelta(seconds=0)).seconds


def get_behavioral_metrics(
    logs_df: pd.DataFrame,
    num_words: int,
    num_chars: int,
) -> dict[str, Any]:
    changes_log = logs_df[logs_df["event_type"] == "change"]
    metrics = {}
    metrics["num_edits"] = len(changes_log)
    metrics["edit_time"] = compute_time(logs_df)
    metrics["edit_time_filtered"] = compute_time(logs_df, datetime.timedelta(minutes=5))
    has_time = metrics["edit_time_filtered"] > 0
    metrics["keys_per_min"] = round(metrics["num_edits"] / (metrics["edit_time_filtered"] / 60), 2) if has_time else 0
    metrics["chars_per_min"] = round(num_chars / (metrics["edit_time_filtered"] / 60), 2) if has_time else 0
    metrics["words_per_min"] = round(num_words / (metrics["edit_time_filtered"] / 60), 2) if has_time else 0
    return metrics


def get_doc_behavioral_data(
    logs_df: pd.DataFrame,
    doc_id: int,
    doc_words: int,
    doc_chars: int,
) -> DocBehavioralMetrics:
    doc_metrics = DocBehavioralMetrics(doc_id=doc_id)
    start_points = list(logs_df[(logs_df["filename"] == f"doc{doc_id}") & (logs_df["event_type"] == "start")].index)

    # If a document was edited twice, the first edited is considered for edit_order
    ordered_docs = list(dict.fromkeys(logs_df["filename"].str.extract(r"doc(\d+)").astype(int).values.T[0]))
    doc_metrics.edit_order = ordered_docs.index(doc_id) + 1

    # Always use latest edit for the document
    start_point = start_points[-1]
    filtered_log = logs_df[start_point:]
    filtered_log = filtered_log[filtered_log["filename"] == f"doc{doc_id}"]
    metrics = get_behavioral_metrics(filtered_log, doc_words, doc_chars)
    doc_metrics.num_edits = metrics["num_edits"]
    doc_metrics.edit_time = metrics["edit_time"]
    doc_metrics.edit_time_filtered = metrics["edit_time_filtered"]
    doc_metrics.keys_per_min = metrics["keys_per_min"]
    doc_metrics.chars_per_min = metrics["chars_per_min"]
    doc_metrics.words_per_min = metrics["words_per_min"]
    return filtered_log, doc_metrics


def add_behavioral_data(
    entry: QE4PEProcessedEntry,
    doc_metrics: DocBehavioralMetrics,
    logs_df: pd.DataFrame,
    segment_id: int,
    ordered_segments: list[int],
) -> QE4PEProcessedEntry:
    entry.doc_num_edits = doc_metrics.num_edits
    entry.doc_edit_order = doc_metrics.edit_order
    entry.doc_edit_time = doc_metrics.edit_time
    entry.doc_edit_time_filtered = doc_metrics.edit_time_filtered
    entry.doc_keys_per_min = doc_metrics.keys_per_min
    entry.doc_chars_per_min = doc_metrics.chars_per_min
    entry.doc_words_per_min = doc_metrics.words_per_min
    entry.segment_edit_order = ordered_segments.index(segment_id) + 1 if segment_id in ordered_segments else -1
    segment_df = logs_df[logs_df["text_id"] == float(segment_id - 1)]
    metrics = get_behavioral_metrics(segment_df, entry.mt_num_words, entry.mt_num_chars)
    entry.segment_num_edits = metrics["num_edits"]
    entry.segment_edit_time = metrics["edit_time"]
    entry.segment_edit_time_filtered = metrics["edit_time_filtered"]
    entry.segment_keys_per_min = metrics["keys_per_min"]
    entry.segment_chars_per_min = metrics["chars_per_min"]
    entry.segment_words_per_min = metrics["words_per_min"]
    entry.num_enter_actions = len(segment_df[segment_df["event_type"] == "enter"])
    entry.remove_highlights = len(segment_df[segment_df["event_type"] == "remove_highlights"]) > 0
    return entry


def extract_highlights(
    text: str,
    tag_pattern: str = r'<(minor|major)>(.*?)</\1>',
    tag_text_pattern: str = r'<(minor|major)>(.*)</\1>',
    tag_severity_pattern: str = r'<(minor|major)>',
) -> list[dict[str, str | int]]:
    curr_txt = text
    highlights = []
    for _ in range(len(list(re.finditer(tag_pattern, text)))):
        match = next(re.finditer(tag_pattern, curr_txt))
        tagged_text = match.group(0)
        tag_text = re.findall(tag_text_pattern, tagged_text)[0][1]
        tag_severity = re.findall(tag_severity_pattern, tagged_text)[0]
        highlights.append({
            "text": tag_text,
            "severity": tag_severity,
            "start": match.span()[0],
            "end": match.span()[0] + len(tag_text)
        })
        curr_txt = curr_txt.replace(tagged_text, tag_text, 1)
    return highlights


def add_texts_and_annotations(
    entry: QE4PEProcessedEntry,
    src_text: str,
    mt_text: str,
    pe_text: str,
    align_words: jiwer.WordOutput,
    align_chars: jiwer.CharacterOutput,
) -> QE4PEProcessedEntry:
    entry.src_text = src_text
    entry.mt_text_highlighted = mt_text
    entry.mt_text = clean_tags(mt_text)
    entry.pe_text = pe_text
    entry.mt_pe_word_aligned = visualize_alignment(align_words)
    entry.mt_pe_char_aligned = visualize_alignment(align_chars)
    entry.highlights = extract_highlights(entry.mt_text_highlighted)
    return entry


def add_quality_metrics(
    entry: QE4PEProcessedEntry,
    doc_metrics: dict[str, float | None | list[dict[str, str | int | float]]],
    segment_id: int,
) -> QE4PEProcessedEntry:
    segment_metrics = doc_metrics[str(segment_id)]
    entry.mt_bleu_max = segment_metrics["mt_bleu_max"]
    entry.mt_bleu_min = segment_metrics["mt_bleu_min"]
    entry.mt_bleu_mean = segment_metrics["mt_bleu_mean"]
    entry.mt_bleu_std = segment_metrics["mt_bleu_std"]
    entry.mt_chrf_max = segment_metrics["mt_chrf_max"]
    entry.mt_chrf_min = segment_metrics["mt_chrf_min"]
    entry.mt_chrf_mean = segment_metrics["mt_chrf_mean"]
    entry.mt_chrf_std = segment_metrics["mt_chrf_std"]
    entry.mt_ter_max = segment_metrics["mt_ter_max"]
    entry.mt_ter_min = segment_metrics["mt_ter_min"]
    entry.mt_ter_mean = segment_metrics["mt_ter_mean"]
    entry.mt_ter_std = segment_metrics["mt_ter_std"]
    entry.mt_comet_max = segment_metrics["mt_comet_max"]
    entry.mt_comet_min = segment_metrics["mt_comet_min"]
    entry.mt_comet_mean = segment_metrics["mt_comet_mean"]
    entry.mt_comet_std = segment_metrics["mt_comet_std"]
    entry.mt_xcomet_qe = segment_metrics["xcomet_mt"]
    entry.mt_xcomet_errors = segment_metrics["xcomet_mt_errors"] if segment_metrics["xcomet_mt_errors"] is not None else []
    entry.pe_bleu_max = segment_metrics["pe_bleu_max"]
    entry.pe_bleu_min = segment_metrics["pe_bleu_min"]
    entry.pe_bleu_mean = segment_metrics["pe_bleu_mean"]
    entry.pe_bleu_std = segment_metrics["pe_bleu_std"]
    entry.pe_chrf_max = segment_metrics["pe_chrf_max"]
    entry.pe_chrf_min = segment_metrics["pe_chrf_min"]
    entry.pe_chrf_mean = segment_metrics["pe_chrf_mean"]
    entry.pe_chrf_std = segment_metrics["pe_chrf_std"]
    entry.pe_ter_max = segment_metrics["pe_ter_max"]
    entry.pe_ter_min = segment_metrics["pe_ter_min"]
    entry.pe_ter_mean = segment_metrics["pe_ter_mean"]
    entry.pe_ter_std = segment_metrics["pe_ter_std"]
    entry.pe_comet_max = segment_metrics["pe_comet_max"]
    entry.pe_comet_min = segment_metrics["pe_comet_min"]
    entry.pe_comet_mean = segment_metrics["pe_comet_mean"]
    entry.pe_comet_std = segment_metrics["pe_comet_std"]
    entry.pe_xcomet_qe = segment_metrics["xcomet_pe"]
    entry.pe_xcomet_errors = segment_metrics["xcomet_pe_errors"] if segment_metrics["xcomet_pe_errors"] is not None else []
    return entry


def process_task_data(
    task_folder_path: Annotated[str, Argument(..., help="Path to the folder containing the task data")],
    output_path: Annotated[str, Argument(..., help="Path to the output file. Uses task_folder_path as default path.")] = None,
):
    """
    Processes the folder containing the task data to create a single dataframe with summarized information about
    translation outputs and editing process. The folder provided to the command should have the following structure:

    \b
    ```
    {{TASK_ID}}/
    ├── inputs/
    │   ├── {{SOURCE_LANG}}-{{TARGET_LANG}}/
    │   │   ├── {{TASK_ID}}_{{SOURCE_LANG}}-{{TARGET_LANG}}_doc1_input.txt
    │   │   ├── {TASK_ID}}_{{SOURCE_LANG}}-{{TARGET_LANG}}_doc2_input.txt
    │   │   └── ... # GroTE input files with tags and ||| source-target separator
    │   └── ... # Other translation directions
    ├── outputs/
    │   ├── {{SOURCE_LANG}}-{{TARGET_LANG}}/
    │   │   ├── logs/
    │   │   │   ├── {{TASK_ID}}_{{SOURCE_LANG}}-{{TARGET_LANG}}_{{TRANSLATOR_ID}}_logs.csv
    │   │   │   └── ... # GroTE logs for every translator (e.g. TRANSLATOR_ID = t1)
    │   │   ├── {{TASK_ID}}_{{SOURCE_LANG}}-{{TARGET_LANG}}_doc1_{{TRANSLATOR_ID}}_output.txt
    │   │   └── ... # GroTE output files (one post-edited segment per line)
    │   └── ... # Other translation directions
    ├── doc_id_map.json # JSON file with mapping between document IDs and their original names (optional)
    └── ... # Other files are ignored
    ```
    """
    logger.info("Processing task data...")
    task_name = os.path.basename(task_folder_path)
    if output_path is None:
        output_path = os.path.join(task_folder_path, f"processed_{task_name}.csv")
    task_config_path = os.path.join(task_folder_path, "processing_config.json")
    if not os.path.exists(task_config_path):
        raise FileNotFoundError(f"Processing configuration file not found: {task_config_path}")
    with open(task_config_path) as f:
        task_config = json.load(f)
    main_task_assignments_path = task_config.get("main_task_assignments_path", None)
    force_highlight_modality = task_config.get("force_highlight_modality", None)
    has_translators_pretask_ids = task_config.get("has_translators_pretask_ids", False)
    has_input_highlight_modality = task_config.get("has_input_highlight_modality", False)
    entries_with_issues = task_config.get("entries_with_issues", [])
    entries_with_added_critical_errors = task_config.get("entries_with_added_critical_errors", [])
    doc_id_map = os.path.join(task_folder_path, "doc_id_map.json")
    input_folder = os.path.join(task_folder_path, "inputs")
    output_folder = os.path.join(task_folder_path, "outputs")
    input_directions = [f for f in os.listdir(input_folder) if not f.startswith('.')]
    output_directions = [f for f in os.listdir(output_folder) if not f.startswith('.')]
    assert input_directions == output_directions, \
        f"Input and output translation directions do not match: {input_directions} != {output_directions}"

    with open(main_task_assignments_path) as f:
        main_task_trans_assignments = json.load(f)

    with open(doc_id_map) as f:
        task_doc_config = json.load(f)
    wmt_doc_map = task_doc_config["map"]
    wmt_doc_config_path = os.path.join(task_folder_path, task_doc_config["original_config"])
    print(wmt_doc_config_path)
    with open(wmt_doc_config_path) as f:
        wmt_doc_category_map = {}
        for line in f:
            line_dic = json.loads(line)
            if line_dic["doc_id"] in wmt_doc_map.values():
                wmt_doc_category_map[line_dic["doc_id"]] = "social" if line_dic["collection_id"] == "general" else line_dic["collection_id"]

    all_entries: list[QE4PEProcessedEntry] = []
    for direction in input_directions:
        print(f"    Processing direction {direction}...")
        direction_assignments = main_task_trans_assignments[direction]
        src_lang, tgt_lang = direction.split("-")
        direction_input_folder = os.path.join(input_folder, direction)
        direction_output_folder = os.path.join(output_folder, direction)
        direction_log_folder = os.path.join(direction_output_folder, "logs")
        direction_metrics_folder = os.path.join(direction_output_folder, "metrics")
        input_files = [fname for fname in os.listdir(direction_input_folder) if fname.endswith(".txt")]
        output_files = [fname for fname in os.listdir(direction_output_folder) if fname.endswith(".txt")]
        file_prefix = f"{task_name}_{src_lang}-{tgt_lang}_doc"
        input_file_suffix = "_input.txt"
        output_file_suffix = "_output.txt"

        # Extract document ids
        split_fname = lambda f, pre, suf: f.lstrip(pre).rstrip(suf).split("_")  # noqa: E731
        doc_ids = sorted({int(split_fname(fname, file_prefix, input_file_suffix)[0]) for fname in input_files})

        # Extract translators identifiers
        translator_ids = sorted({"_".join(split_fname(fname, file_prefix, output_file_suffix)[1:]) for fname in output_files})

        if has_translators_pretask_ids:
            translator_main_ids, translator_modalities = zip(
                *[(direction_assignments[translator_id]['alias'], direction_assignments[translator_id]['modality'])
                for translator_id in translator_ids], strict=True
            )
            translator_pre_ids = translator_ids
        else:
            # Reverse lookup for main task assignments
            reverse_assignments = {translator_dic['alias']: translator_pre_id for translator_pre_id, translator_dic in direction_assignments.items()}
            translator_pre_ids = [reverse_assignments[id] for id in translator_ids]
            translator_main_ids = translator_ids
            translator_modalities = [direction_assignments[translator_id]['modality'] for translator_id in translator_pre_ids]

        for file_id, main_id, pre_id, modality in zip(translator_ids, translator_main_ids, translator_pre_ids, translator_modalities, strict=True):
            translator_log_fname = f"{task_name}_{src_lang}-{tgt_lang}_{file_id}_logs.csv"
            translator_log_path = os.path.join(direction_log_folder, translator_log_fname)
            try:
                translator_log = pd.read_csv(translator_log_path)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Logs file not found: {translator_log_path}") from err
            except pd.errors.ParserError as err:
                raise pd.errors.ParserError(f"Error parsing logs file: {translator_log_path}") from err
            translator_log["time"] = pd.to_datetime(translator_log["time"])
            translator_metrics_fname = f"{task_name}_{src_lang}-{tgt_lang}_{file_id}_metrics.json"
            translator_metrics_path = os.path.join(direction_metrics_folder, translator_metrics_fname)
            if not os.path.exists(translator_metrics_path):
                print(f"Metrics file not found: {translator_metrics_path}")
                translator_metrics = None
            else:
                with open(translator_metrics_path) as f:
                    translator_metrics = json.load(f)

            segment_id = 1
            for doc_id in doc_ids:
                curr_doc_metrics = None
                if translator_metrics is not None:
                    curr_doc_metrics = translator_metrics[str(doc_id)]
                if has_input_highlight_modality:
                    doc_input_fname = f"{file_prefix}{doc_id}_{modality}{input_file_suffix}"
                else:
                    doc_input_fname = f"{file_prefix}{doc_id}{input_file_suffix}"
                doc_input_path = os.path.join(direction_input_folder, doc_input_fname)
                with open(doc_input_path) as f:
                    doc_input = [l.strip() for l in f.readlines()]
                doc_output_fname = f"{file_prefix}{doc_id}_{file_id}{output_file_suffix}"
                doc_output_path = os.path.join(direction_output_folder, doc_output_fname)
                with open(doc_output_path) as f:
                    doc_output = [l.strip() for l in f.readlines()]
                assert len(doc_input) == len(doc_output), \
                    f"Input and output lengths do not match for {doc_input_fname} and {doc_output_fname}: {len(doc_input)} != {len(doc_output)}"

                # Precompute tot words/chars for the full document
                doc_mt_words = 0
                doc_mt_chars = 0
                for src_mt_text in doc_input:
                    _, mt_text = src_mt_text.split(" ||| ")
                    mt_text = clean_tags(mt_text)
                    doc_mt_words += len(mt_text.split())
                    doc_mt_chars += len(mt_text)
                doc_logs, doc_behavioral_data = get_doc_behavioral_data(translator_log, doc_id, doc_mt_words, doc_mt_chars)
                ordered_segments = [x+1 for x in list(dict.fromkeys(doc_logs["text_id"].dropna().astype(int).values))]
                for segment_in_doc_id, (src_mt_text, pe_text) in enumerate(zip(doc_input, doc_output, strict=True), start=1):
                    src_text, mt_text = src_mt_text.split(" ||| ")
                    entry_issue = None
                    entry_critical_error = None
                    for ei in entries_with_issues:
                        if ei["doc_id"] == doc_id and ei["segment_in_doc_id"] == segment_in_doc_id:
                            entry_issue = ei
                            break
                    for ece in entries_with_added_critical_errors:
                        if ece["doc_id"] == doc_id and ece["segment_in_doc_id"] == segment_in_doc_id:
                            entry_critical_error = ece
                            break
                    entry = QE4PEProcessedEntry()
                    entry = add_identification_info(
                        entry=entry,
                        task_name=task_name,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        doc_id=doc_id,
                        segment_in_doc_id=segment_in_doc_id,
                        segment_id=segment_id,
                        current_translator_id=file_id,
                        pretask_translator_id=pre_id,
                        main_translator_id=main_id,
                        modality=modality,
                        wmt_doc_map=wmt_doc_map,
                        wmt_doc_category_map=wmt_doc_category_map,
                        force_highlight_modality=force_highlight_modality,
                        entry_issue=entry_issue,
                        entry_critical_error=entry_critical_error,
                    )
                    entry = add_data_stats(entry=entry, src_text=src_text, mt_text=mt_text, pe_text=pe_text)
                    align_words = jiwer.process_words(clean_tags(mt_text), pe_text)
                    align_chars = jiwer.process_characters(clean_tags(mt_text), pe_text)
                    entry = add_edits_stats(entry=entry, align_words=align_words, align_chars=align_chars)
                    entry = add_behavioral_data(
                        entry=entry,
                        doc_metrics=doc_behavioral_data,
                        logs_df=doc_logs,
                        segment_id=segment_in_doc_id,
                        ordered_segments=ordered_segments,
                    )
                    entry = add_texts_and_annotations(
                        entry=entry,
                        src_text=src_text,
                        mt_text=mt_text,
                        pe_text=pe_text,
                        align_words=align_words,
                        align_chars=align_chars
                    )
                    if translator_metrics is not None:
                        entry = add_quality_metrics(entry=entry, doc_metrics=curr_doc_metrics, segment_id=segment_in_doc_id)
                    all_entries.append(entry)
                    segment_id += 1

    # Save processed entries to CSV
    df = pd.DataFrame([entry.__dict__ for entry in all_entries])

    if "qa_path" in task_doc_config:
        qa_path = os.path.join(task_folder_path, task_doc_config["qa_path"])
        mqm_df = pd.read_csv(qa_path)
        df = df.merge(
            mqm_df,
            on=['doc_id', 'segment_in_doc_id', 'tgt_lang', 'translator_main_id'],
            how='left'
        )
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    logger.info("Done!")


def process_task_data_callback(verbose: Annotated[bool, Option(..., help="Increase verbosity")] = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
