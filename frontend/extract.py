import json
import os

log_file = r'C:\Users\loq\.gemini\antigravity-ide\brain\4e654601-ca1d-422d-9c51-23c645f5da7f\.system_generated\logs\transcript.jsonl'
contents = []

for line in open(log_file, encoding='utf-8'):
    try:
        step = json.loads(line)
        if step.get('type') == 'PLANNER_RESPONSE' and step.get('tool_calls'):
            for tc in step['tool_calls']:
                if tc.get('name') == 'write_to_file' and 'styles.css' in str(tc.get('args', {}).get('TargetFile', '')):
                    contents.append(tc['args'].get('CodeContent', ''))
    except:
        pass
        
for i, c in enumerate(contents):
    val = c
    if isinstance(c, str) and c.startswith('"') and c.endswith('"'):
        try:
            val = json.loads(c)
        except:
            pass
    with open(f'styles_{i}.css', 'w', encoding='utf-8') as f:
        f.write(val)
print(f'Wrote {len(contents)} css files.')
