"""Integration checks against a running local API; optional failing image path."""
import sys,json
from pathlib import Path
import httpx

root=Path(__file__).resolve().parents[1]
health=httpx.get('http://127.0.0.1:8000/api/health').json()
assert health['detector_ready'] and not health['vehicle_detector_ready'],health
cases=[(root/'tests/three-vehicle-composite.jpg',3,None)]
if len(sys.argv)>1:cases.append((Path(sys.argv[1]),2,'GJ27DS4837'))
summary=[]
for path,minimum,text in cases:
    response=httpx.post('http://127.0.0.1:8000/api/detect/image',files={'image':(path.name,path.read_bytes(),'image/jpeg')},data={'profile':'detailed'},timeout=60)
    response.raise_for_status();data=response.json()
    assert data['vehicles']==[]
    assert data['search_strategy']=='whole_frame_and_overlapping_tiles'
    readings=[d['normalized_text'] for d in data['detections'] if d['normalized_text']]
    assert len(readings)>=minimum,(path.name,readings)
    if text:assert text in readings,readings
    summary.append(dict(case='user_scooter_photo' if text else 'three_vehicle_composite',regions=len(data['detections']),text_readings=len(readings),vehicle_inference=False,processing_time_ms=data['processing_time_ms']))
print(json.dumps(summary,indent=2))
(root/'tests/plate-first-regression.json').write_text(json.dumps(summary,indent=2))
