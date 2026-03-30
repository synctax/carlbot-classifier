# Creates unclassified training data from raw discord chats
#
#
# Input headers:
# Date | Username | User tag | Content | Mentions | Class*
# *optional
#
# Output headers:
# Text | Answer
#
# Output:
# - Replaces raw mentions with "@<user>" in message content
# - includes <context>+1 number of messages in each sample

import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser(
    description="Cleans raw discord chat data for labelling"
)

parser.add_argument("input", help="The path of the input CSV file")
parser.add_argument("--out", help="The path of the output csv file")
parser.add_argument(
    "--context",
    type=int,
    help="Number of previous messages to include in each sample"
)

args = parser.parse_args()
out_file = args.out or f"{args.input[:-4]}_cleaned.csv"


df = pd.read_csv(args.input)
df = df.loc[:, ~df.columns.duplicated()].copy()
col_map = {c.lower(): c for c in df}


def get_col(name):
    return col_map.get(name.lower()) or name


date_col = get_col("date")
class_col = get_col("class")
username_col = get_col("username")
tag_col = get_col("user tag")
content_col = get_col("content")
mentions_col = get_col("mentions")


if class_col.lower() not in [c.lower() for c in df.columns]:
    df[class_col] = None

df = df.dropna(subset=[content_col])
df = df[df[content_col].str.strip() != ""]
df = df[df[tag_col].str.strip() == "#0"]


def replace_mentions(row):
    text = row[content_col]
    mentions = row[mentions_col]

    usernames = [u.strip() for u in mentions.split(',')]

    mention_pattern = r"<@!?\d+>"

    def replace_next_username(match):
        if usernames:
            return f"@{usernames.pop().split('#')[0]}"
        else:
            return match.group(0)

    return re.sub(mention_pattern, replace_next_username, str(text))


mention_mask = (df[mentions_col]).notna() & (
    df[mentions_col].astype(str).str.strip() != "")

df.loc[mention_mask, content_col] = df[mention_mask].apply(
    replace_mentions, axis=1)

text_col = "Text"
answer_col = "Answer"


df[content_col] = df[username_col] + ": " + df[content_col]

context_cols = []
for i in range(args.context, 0, -1):
    col_name = f"prev_msg_{i}"
    df[col_name] = df[content_col].shift(i).fillna("")
    context_cols.append(col_name)

df[text_col] = df[context_cols].agg('\n'.join, axis=1) + '\n' + df[content_col]

df[answer_col] = df[class_col]

out_df = df[['Text', 'Answer']]
out_df.to_csv(out_file, index=False, encoding='utf-8-sig')
