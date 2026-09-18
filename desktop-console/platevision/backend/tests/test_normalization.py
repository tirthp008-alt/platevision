from app.services.normalization import normalize_plate
def test_indian_plate(): assert normalize_plate('GJ 01-AB 1234') == ('GJ01AB1234','GJ 01 AB 1234','valid')
def test_uncertain(): assert normalize_plate('abc')[2] == 'uncertain'

def test_bharat_series():
    assert normalize_plate('22 BH 1234 AA')[2] == 'valid'
