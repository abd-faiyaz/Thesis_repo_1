# Excerpt from Shared_pipeline_Files/tools/jsonl_to_json.py
def merge(jsonl_path, out_path):
    scans, sessions = load_records(jsonl_path)
    device = scans[0].get("device", {}) if scans else {}
    payload = {"device": device, "scans": scans}
    if sessions:
        payload["sessions"] = sessions
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return len(scans), len(sessions)
