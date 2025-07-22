import json
import argparse
import csv
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file as list of dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a string."""
    if text is None:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return {}
    return {}


def get_last_message(item: Dict[str, Any]) -> str:
    """Get the last assistant message from conversation."""
    conversations = item.get('conversations') or item.get('messages') or []
    if isinstance(conversations, list) and conversations:
        last = conversations[-1]
        if isinstance(last, dict):
            return last.get('value') or last.get('content', '')
        return str(last)
    return ''


def evaluate(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]):
    fields = ['team_name', 'primary_player_number', 'label', 'result']
    length = min(len(preds), len(gts))
    results = []
    hits = 0
    for i in range(length):
        pred_content = get_last_message(preds[i])
        pred_json = extract_json(pred_content)

        hit = True
        row = {'index': i}
        for f in fields:
            p_val = str(pred_json.get(f, '')).strip()
            g_val = str(gts[i].get(f, '')).strip()
            row[f'pred_{f}'] = p_val
            row[f'gt_{f}'] = g_val
            if p_val != g_val:
                hit = False
        row['hit'] = int(hit)
        hits += int(hit)
        results.append(row)
        print(f'Item {i}: {row["hit"]}')

    overall = hits / length if length > 0 else 0.0
    print(f'Overall accuracy: {overall:.4f} ({hits}/{length})')
    return results, overall


def main():
    parser = argparse.ArgumentParser(description='Evaluate basketball predictions')
    parser.add_argument('--pred-file', required=True, help='Model predictions JSON file')
    parser.add_argument('--gt-file', required=True, help='Ground truth JSON file')
    parser.add_argument('--output-file', help='Optional path to save evaluation CSV')
    args = parser.parse_args()

    preds = load_json(args.pred_file)
    gts = load_json(args.gt_file)

    results, overall = evaluate(preds, gts)

    if args.output_file and results:
        with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {args.output_file}')


if __name__ == '__main__':
    main()
