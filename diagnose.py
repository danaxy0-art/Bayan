import pandas as pd
import json

df = pd.read_csv("data/search/bayan_cases.csv")
case_lookup = dict(zip(df["case_id"], df["case_text"]))
case_topic = dict(zip(df["case_id"], df["topic"]))

queries = []
with open("data/search/bayan_queries.jsonl", encoding="utf-8") as f:
    for line in f:
        queries.append(json.loads(line))

answerable = [q for q in queries if not q["no_answer"]]

print(f"Checking {len(answerable)} answerable queries\n")

topic_match_count = 0
missing_ids_count = 0

for q in answerable[:20]:  # print first 20 in detail
    print(f"Query {q['query_id']} [{q['topic']}]: {q['query']}")
    for cid in q["relevant_case_ids"]:
        if cid not in case_lookup:
            print(f"  {cid} -> MISSING FROM CSV")
            continue
        rel_topic = case_topic[cid]
        match = "SAME TOPIC" if rel_topic == q["topic"] else "DIFFERENT TOPIC"
        print(f"  {cid} [{rel_topic}] {match}: {case_lookup[cid][:60]}")
    print()

# Aggregate stats across ALL answerable queries
same_topic_total = 0
diff_topic_total = 0
missing_total = 0
total_refs = 0

for q in answerable:
    for cid in q["relevant_case_ids"]:
        total_refs += 1
        if cid not in case_lookup:
            missing_total += 1
        elif case_topic[cid] == q["topic"]:
            same_topic_total += 1
        else:
            diff_topic_total += 1

print("=" * 60)
print("AGGREGATE STATS across all", len(answerable), "answerable queries")
print("=" * 60)
print(f"Total relevant_case_id references: {total_refs}")
print(f"  Same topic as query:      {same_topic_total} ({same_topic_total/total_refs*100:.1f}%)")
print(f"  Different topic as query: {diff_topic_total} ({diff_topic_total/total_refs*100:.1f}%)")
print(f"  Missing from CSV:         {missing_total} ({missing_total/total_refs*100:.1f}%)")