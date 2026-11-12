"""Finish packaging the existing verification queue while the user is away."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

base=Path(r'C:\Users\35884\Documents\Spacecraft')
audit=base/'V3E_50K_EXECUTION_20260928.json'
status=base/'V3E_50K_AUTOPACK_STATUS_20260929.json'
def save(data):
    data['updated_at']=datetime.now().astimezone().isoformat()
    status.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
save({'status':'waiting_for_existing_queue','training_not_modified':True})
while True:
    try:
        data=json.loads(audit.read_text())
    except (OSError,json.JSONDecodeError):
        time.sleep(30)
        continue
    if 'completed_at' in data:
        save({'status':'packaging','execution_completed_at':data['completed_at']})
        cmd=[sys.executable,'-u','-B',str(base/'package_v3e_50k_check_20260928.py')]
        proc=subprocess.run(cmd,cwd=r'D:\py\DRL2_v3e')
        save({'status':'completed' if proc.returncode==0 else 'failed','exit_code':proc.returncode,
              'report':str(base/'V3E_50K_DELIVERY_20260928'/'REPORT.md'),
              'archive':str(base/'V3E_50K_DELIVERY_20260928.zip'),'training_not_modified':True})
        break
    time.sleep(30)
