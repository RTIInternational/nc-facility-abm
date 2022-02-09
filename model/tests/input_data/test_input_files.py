import src.data_input as di


def test_county_hospital_distances():
    """Test the county hospital distance file to make sure each county and each hospital are there
    make sure all distances are non-negative
    """
    hospitals = di.hospitals()
    ch = di.county_hospital_distances()
    hospital_names = hospitals.Name.values

    for county in range(1, 201, 2):
        assert county in ch.keys()
        assert len(ch[county]) >= len(hospital_names)


def test_length_of_stay_files():
    nh = di.nh_los()
    all([v >= 0 for v in nh])


def test_id_files():
    """Test nh_ids.csv, lt_ids.csv, hospital_ids.csv"""
    hs = di.hospitals()
    nh = di.nursing_homes()
    lt = di.ltachs()

    assert all(hs["ICU Beds"] >= 0)
    assert all(hs["Beds"] < 10000)
    assert all(hs["Beds"] >= hs["ICU Beds"])

    assert not any(nh.lat.isna())
    assert not any(nh.lon.isna())
    assert all(nh.Beds > 0)

    assert all(lt.Beds > 0)
