python3 -c "
import json
with open('models_cache.json') as f:
    data = json.load(f)
version = data.pop('_version', 0)
count = 0
for key, val in data.items():
    if isinstance(val, dict):
        val['file_size_bytes'] = 0
        count += 1
data['_version'] = version
with open('models_cache.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'{count} Einträge zum Re-Scan markiert')
"
