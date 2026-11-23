
"""
DataLens ML-07: Association Rule Mining

Algorithms:
- Apriori
- FP-Growth

Input formats:
- Transaction: one row per transaction, with an item-list column.
- Long format: transaction ID and one item per row.
- One-hot: one row per transaction, with boolean or 0/1 item columns.

Metrics:
- Support
- Confidence
- Lift
- Leverage
- Conviction
"""

from io import BytesIO
import inspect

import joblib
import pandas as pd

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


# =========================================================
# ALGORITHM LIBRARY
# =========================================================

def get_association_algorithms():
    return {
        "Apriori": {
            "description": (
                "Finds frequent itemsets using candidate generation "
                "and the Apriori property."
            ),
            "type": "Candidate-based"
        },
        "FP-Growth": {
            "description": (
                "Finds frequent itemsets using an FP-tree, avoiding "
                "explicit candidate generation."
            ),
            "type": "Tree-based"
        }
    }


# =========================================================
# TRANSACTION DATA PREPARATION
# =========================================================

def prepare_transaction_data(
    df,
    item_column,
    transaction_column=None,
    separator=","
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Please provide a non-empty dataset.")

    if item_column not in df.columns:
        raise ValueError("The selected item column does not exist.")

    if transaction_column and transaction_column not in df.columns:
        raise ValueError("The selected transaction ID column does not exist.")

    if not separator:
        raise ValueError("Please provide a valid item separator.")

    def parse_items(value):
        if pd.isna(value):
            return []

        if isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = str(value).split(separator)

        cleaned = []
        seen = set()

        for item in values:
            item = str(item).strip()

            if item and item not in seen:
                cleaned.append(item)
                seen.add(item)

        return cleaned

    transactions = []

    if transaction_column:
        grouped = df.groupby(
            transaction_column,
            sort=False,
            dropna=False
        )

        for _, group in grouped:
            items = []

            for value in group[item_column]:
                items.extend(parse_items(value))

            items = list(dict.fromkeys(items))

            if items:
                transactions.append(items)

    else:
        for value in df[item_column]:
            items = parse_items(value)

            if items:
                transactions.append(items)

    if len(transactions) < 2:
        raise ValueError(
            "At least two non-empty transactions are required."
        )

    return transactions


# =========================================================
# ONE-HOT DATA PREPARATION
# =========================================================

def prepare_onehot_data(df, excluded_columns=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Please provide a non-empty dataset.")

    excluded_columns = excluded_columns or []

    missing = [
        column for column in excluded_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Unknown excluded columns: {missing}")

    data = df.drop(columns=excluded_columns).copy()

    if data.empty:
        raise ValueError("No item columns remain after exclusions.")

    for column in data.columns:
        series = data[column]
        non_missing = series.dropna()

        if non_missing.empty:
            data[column] = False
            continue

        if pd.api.types.is_bool_dtype(series):
            data[column] = series.fillna(False).astype(bool)
            continue

        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().sum() == len(non_missing):
            unique_values = set(numeric.dropna().unique())

            if unique_values.issubset({0, 1}):
                data[column] = numeric.fillna(0).astype(bool)
                continue

        normalized = series.astype(str).str.strip().str.lower()

        mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
            "1": True,
            "0": False
        }

        if normalized.isin(mapping).all():
            data[column] = normalized.map(mapping).astype(bool)
        else:
            raise ValueError(
                f"Column '{column}' is not a valid one-hot column. "
                "Use boolean or 0/1 values, or exclude it."
            )

    # Remove item columns that never occur.
    data = data.loc[:, data.any(axis=0)]

    # Remove transactions containing no items.
    data = data.loc[data.any(axis=1)]

    if data.empty or data.shape[1] == 0:
        raise ValueError("No positive item occurrences were found.")

    if len(data) < 2:
        raise ValueError("At least two non-empty transactions are required.")

    return data


# =========================================================
# TRANSACTION ENCODING
# =========================================================

def encode_transactions(transactions):
    if not transactions:
        raise ValueError("No transactions were supplied.")

    encoder = TransactionEncoder()

    encoded = encoder.fit(transactions).transform(transactions)

    basket = pd.DataFrame(
        encoded,
        columns=encoder.columns_
    )

    if basket.empty or basket.shape[1] == 0:
        raise ValueError("No usable items were found.")

    return basket, encoder


# =========================================================
# ASSOCIATION RULE GENERATION
# =========================================================

def _generate_rules(frequent_itemsets, transaction_count, min_confidence):
    """
    Supports mlxtend versions where association_rules either
    requires or does not require the num_itemsets argument.
    """

    parameters = inspect.signature(association_rules).parameters

    kwargs = {
        "metric": "confidence",
        "min_threshold": min_confidence
    }

    if "num_itemsets" in parameters:
        kwargs["num_itemsets"] = transaction_count

    return association_rules(
        frequent_itemsets,
        **kwargs
    )


# =========================================================
# MAIN MINING ENGINE
# =========================================================

def run_association_mining(
    data,
    algorithm="Apriori",
    input_format="Transaction",
    item_column=None,
    transaction_column=None,
    separator=",",
    excluded_columns=None,
    min_support=0.05,
    min_confidence=0.5,
    min_lift=1.0,
    max_len=None
):
    if algorithm not in get_association_algorithms():
        raise ValueError("Unsupported association algorithm.")

    if not 0 < min_support <= 1:
        raise ValueError("Minimum support must be greater than 0 and at most 1.")

    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum confidence must be between 0 and 1.")

    if min_lift < 0:
        raise ValueError("Minimum lift cannot be negative.")

    if max_len is not None and max_len < 1:
        raise ValueError("Maximum itemset length must be at least 1.")

    encoder = None
    transactions = None

    if input_format == "Transaction":
        if not item_column:
            raise ValueError("Select the column containing transaction items.")

        transactions = prepare_transaction_data(
            data,
            item_column=item_column,
            transaction_column=transaction_column,
            separator=separator
        )

        basket, encoder = encode_transactions(transactions)

    elif input_format == "One-hot":
        basket = prepare_onehot_data(
            data,
            excluded_columns=excluded_columns
        )

    else:
        raise ValueError(
            "Input format must be Transaction or One-hot."
        )

    transaction_count = len(basket)

    # -----------------------------------------------------
    # FREQUENT ITEMSETS
    # -----------------------------------------------------

    if algorithm == "Apriori":
        frequent_itemsets = apriori(
            basket,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
            low_memory=True
        )
    else:
        frequent_itemsets = fpgrowth(
            basket,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len
        )

    if frequent_itemsets.empty:
        raise ValueError(
            "No frequent itemsets found. Try reducing minimum support."
        )

    frequent_itemsets = frequent_itemsets.copy()

    frequent_itemsets["Itemset Length"] = (
        frequent_itemsets["itemsets"].apply(len)
    )

    frequent_itemsets["Items"] = (
        frequent_itemsets["itemsets"].apply(
            lambda values: ", ".join(sorted(map(str, values)))
        )
    )

    frequent_itemsets = frequent_itemsets.sort_values(
        ["support", "Itemset Length"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # ASSOCIATION RULES
    # -----------------------------------------------------

    rules = _generate_rules(
        frequent_itemsets[["support", "itemsets"]],
        transaction_count,
        min_confidence
    )

    if not rules.empty:
        rules = rules.loc[rules["lift"] >= min_lift].copy()

        rules["Antecedents"] = rules["antecedents"].apply(
            lambda values: ", ".join(sorted(map(str, values)))
        )

        rules["Consequents"] = rules["consequents"].apply(
            lambda values: ", ".join(sorted(map(str, values)))
        )

        rules["Rule"] = (
            rules["Antecedents"]
            + " → "
            + rules["Consequents"]
        )

        rules = rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=False
        ).reset_index(drop=True)

        rule_output = rules[
            [
                "Antecedents",
                "Consequents",
                "Rule",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction"
            ]
        ].rename(columns={
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
            "leverage": "Leverage",
            "conviction": "Conviction"
        })

    else:
        rule_output = pd.DataFrame(columns=[
            "Antecedents",
            "Consequents",
            "Rule",
            "Support",
            "Confidence",
            "Lift",
            "Leverage",
            "Conviction"
        ])

    itemset_output = frequent_itemsets[
        ["Items", "support", "Itemset Length"]
    ].rename(columns={
        "Items": "Itemset",
        "support": "Support"
    })

    summary = {
        "algorithm": algorithm,
        "input_format": input_format,
        "transaction_count": int(transaction_count),
        "item_count": int(basket.shape[1]),
        "frequent_itemset_count": int(len(frequent_itemsets)),
        "association_rule_count": int(len(rule_output)),
        "min_support": float(min_support),
        "min_confidence": float(min_confidence),
        "min_lift": float(min_lift)
    }

    return {
        "algorithm": algorithm,
        "encoder": encoder,
        "transactions": transactions,
        "basket_data": basket,
        "frequent_itemsets": frequent_itemsets,
        "itemsets_output": itemset_output,
        "rules": rules,
        "rules_output": rule_output,
        "summary": summary
    }


# =========================================================
# RULE FILTERING
# =========================================================

def filter_association_rules(
    rules_df,
    minimum_support=0.0,
    minimum_confidence=0.0,
    minimum_lift=0.0
):
    if rules_df is None or rules_df.empty:
        return pd.DataFrame()

    filtered = rules_df.loc[
        (rules_df["Support"] >= minimum_support)
        & (rules_df["Confidence"] >= minimum_confidence)
        & (rules_df["Lift"] >= minimum_lift)
    ].copy()

    return filtered.sort_values(
        ["Lift", "Confidence", "Support"],
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# EXPORT FUNCTIONS
# =========================================================

def export_frequent_itemsets(results):
    return results["itemsets_output"].to_csv(
        index=False
    ).encode("utf-8")


def export_association_rules(results):
    return results["rules_output"].to_csv(
        index=False
    ).encode("utf-8")


def export_association_model(results):
    export_object = {
        "algorithm": results["algorithm"],
        "input_format": results["summary"]["input_format"],
        "encoder": results["encoder"],
        "frequent_itemsets": results["frequent_itemsets"],
        "rules": results["rules"],
        "summary": results["summary"]
    }

    buffer = BytesIO()
    joblib.dump(export_object, buffer)
    buffer.seek(0)

    return buffer.getvalue()
