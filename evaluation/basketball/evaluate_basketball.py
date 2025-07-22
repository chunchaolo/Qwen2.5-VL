import json
import argparse
import csv
from typing import List, Dict, Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file as a list of dictionaries."""
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


def get_first_human_message(item: Dict[str, Any]) -> str:
    """Return the text of the first human/user message."""
    convs = item.get('conversations') or item.get('messages') or []
    if not isinstance(convs, list):
        return ''
    for msg in convs:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role') or msg.get('from')
        if role in {'human', 'user'}:
            return msg.get('value') or msg.get('content', '')
    if convs and isinstance(convs[0], dict):
        return convs[0].get('value') or convs[0].get('content', '')
    return ''


def build_model(model_path: str):
    """Load Qwen model and processor on GPU if available."""
    if any(k in model_path.lower() for k in ['2.5', '2_5', 'qwen25']):
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path)
    model = model_cls.from_pretrained(model_path, torch_dtype='auto', device_map='auto')
    model.eval()
    return model, processor


def predict(item: Dict[str, Any], model, processor, device: str) -> str:
    """Generate model output for one item."""
    video = item.get('video') or item.get('video_path') or item.get('video_url')
    if not video:
        return ''
    prompt_text = get_first_human_message(item)
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'video', 'video': video},
            {'type': 'text', 'text': prompt_text}
        ]
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors='pt', **video_kwargs)
    inputs = inputs.to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=256)
    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    output = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output[0]


def run_inference(gts: List[Dict[str, Any]], model_path: str, device: str) -> List[Dict[str, Any]]:
    model, processor = build_model(model_path)
    preds = []
    for item in gts:
        text = predict(item, model, processor, device)
        preds.append({'conversations': [{'role': 'assistant', 'content': text}]})
    return preds


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
    parser = argparse.ArgumentParser(description='Run basketball video evaluation')
    parser.add_argument('--model-path', required=True, help='Path to the Qwen model')
    parser.add_argument('--gt-file', required=True, help='Ground truth JSON file')
    parser.add_argument('--pred-file', required=True, help='Where to save predictions JSON')
    parser.add_argument('--output-file', help='Optional CSV with evaluation details')
    parser.add_argument('--device', default='cuda', help='Device for model inference')
    args = parser.parse_args()

    gts = load_json(args.gt_file)
    preds = run_inference(gts, args.model_path, args.device)

    with open(args.pred_file, 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f'Predictions saved to {args.pred_file}')

    results, overall = evaluate(preds, gts)

    if args.output_file and results:
        with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {args.output_file}')


if __name__ == '__main__':
    main()
