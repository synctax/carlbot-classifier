# Uses an llm to automatically label data from an input csv
#
# will not label rows that have already been labelled


from openrouter import OpenRouter
import pandas as pd
import os
import argparse

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "no key given"
MODEL_NAME = 'google/gemini-3-flash-preview'
SYSTEM_PROMPT = """
    Role: You are an Intent Classifier for a Discord chat. You are aggressively biased toward capturing messages that initiate or confirm a group activity (gaming, watching, or calling).

    Task: Classify the intent of the VERY LAST MESSAGE in the provided snippet.

    Classification Criteria:
    Respond '1' (Yes) if the LAST MESSAGE is:
    1. An Invitation/Initiation: Actively inviting others to join or asking if anyone is available (e.g., "CS?", "Who's on?", "hop on?", "anyone playing?", "val?").
    2. A Commitment/Response: Confirming they are joining or agreeing to an invitation (e.g., "bet," "down," "omw," "in 5," "booting up").
    3. A Status Update (Joining): Signaling they are currently transitioning into the activity (e.g., "joining now," "pulling up").

    Respond '0' (No) if the LAST MESSAGE is:
    1. Already Present: The user indicates they are ALREADY in the call or game (e.g., "I'm here," "I'm in VC," "I'm already playing").
    2. General Discussion: Chatting about game mechanics, Steam Decks, or technical issues without an intent to play.
    3. Future Planning: Plans for "later," "tonight," or "tomorrow."
    4. Declining: Saying "no," "can't," or "maybe later."

    CRITICAL RULES:
    - Focus ONLY on the intent of the final user message. Use previous messages ONLY to understand what the final user is replying to.
    - Do NOT label '1' if the final user is already active in the activity.
    - If the last message is a question inviting people to play, label '1' (be aggressive with invitations).

    Response Format:
    [1 or 0]

    ----- BEGIN MESSAGES -----
"""

parser = argparse.ArgumentParser(
    description="Automatially labels input training data using an LLM"
)

parser.add_argument("file", help="Path to training data file")
parser.add_argument("--out", help="Path to write labelled file")

args = parser.parse_args()
outpath = args.out or f"{args.file[:-4]}_labelled.csv"

df = pd.read_csv(args.file)
text_col = "Text"
answer_col = "Answer"

df[answer_col] = df[answer_col].astype(str)


def safe_save(df):
    temp_path = f"{outpath}.tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, outpath)


if answer_col not in df.columns:
    df[answer_col] = None

unanswered_mask = df[answer_col].isna() | (
    df[answer_col].astype(str).str.strip().isin(["", "nan"]))

indices_to_label = df[unanswered_mask].index

print(f"Total rows to label: {len(indices_to_label)}")

with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
    try:
        for count, idx in enumerate(indices_to_label, 1):
            row = df.loc[idx]
            text = row[text_col]
            last_line = text.split('\n')[-1]

            response = client.chat.send(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": f"{SYSTEM_PROMPT}{text}"}
                ]
            )

            answer = response.choices[0].message.content
            print(f"[{count}/{len(indices_to_label)
                              }] Got answer: {answer} For Line: {last_line}")

            df.at[idx, answer_col] = f"{answer}"

            if count % 10 == 0:
                safe_save(df)
    except KeyboardInterrupt:
        print("\nPerforming final save...")
    finally:
        safe_save(df)
        print(f"Progress saved to {outpath}")
        safe_save(df)
