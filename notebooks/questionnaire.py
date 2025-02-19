# %%
import datasets
import numpy as np

data = (
    list(datasets.load_dataset("gsarti/qe4pe", "posttask_highlight_questionnaire")["train"]) +
    list(datasets.load_dataset("gsarti/qe4pe", "posttask_no_highlight_questionnaire")["train"])
)

# %%
keys = [
    ("mt_quality", "MT Quality", True),
    ("interface_statement_4", "Challenging to translate", True),
    ("highlight_accuracy", "Highlights accurate", False),
    ("highlight_usefulness", "Highlights useful", False),
    ("highlight_statement_2", "Highlights improved quality", False),
    ("highlight_statement_4", "Highlights required more effort", False),
    ("highlight_statement_5", "Highlights influenced editing", False),
    ("highlight_statement_6", "Highlights helped identify errors", False),
]

# order:
# no highlight - blue
# oracle - green
# supervised - pink
# unsupervised - orange
for key, key_name, allow_no_highlight in keys:
    data_out = []
    for modality in ["no_highlight", "oracle", "supervised", "unsupervised"]:
        if modality == "no_highlight" and not allow_no_highlight:
            data_out.append(0)
            continue
        data_out.append(np.average([d[key] for d in data if d["main_task_modality"] == modality and key in d]))

    print(f"{key_name:>40}", r"\modblocks{", end="")
    print(*[f"{x:.2f}" for x in data_out], sep=r"}{", end="}\n")

# %%

# TODO: all questionnaire results in table in appendix