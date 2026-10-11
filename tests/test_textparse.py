import math

from vanagni.textparse import parse


def test_english_bulletin():
    text = ("IMD MC Dehradun | Range forecast: Kalsi | Issued 02 May 2026 17:30 IST\n"
            "Observed Tmax 34.6C. Dry spell enters day 19. Max temp expected to touch "
            "37 deg C tomorrow. Afternoon humidity: 18-44%. Winds north-westerly at "
            "21 km/h, gusting to 34 km/h. No rain expected. Forest fire danger: VERY HIGH.")
    ex = parse(text)
    assert ex["obs_tmax"] == 34.6
    assert ex["fc_tmax"] == 37
    assert ex["fc_rh"] == 18
    assert ex["fc_wind"] == 21 and ex["fc_gust"] == 34
    assert ex["dry_spell"] == 19
    assert ex["fc_rain"] == 0
    assert ex["danger"] == 3          # "very high", not "high"


def test_hindi_bulletin():
    text = ("मौसम बुलेटिन – मसूरी वन क्षेत्र, 02.05.2026\n"
            "आज अधिकतम तापमान 29.4°C रहा। पिछले 12 दिनों से कोई वर्षा नहीं। "
            "कल अधिकतम तापमान लगभग 31°C रहने की संभावना। न्यूनतम सापेक्ष आर्द्रता 24% तक। "
            "हवा 17 किमी/घंटा की गति से। मौसम शुष्क रहेगा। वनाग्नि खतरा स्तर: अति उच्च।")
    ex = parse(text)
    assert ex["obs_tmax"] == 29.4 and ex["fc_tmax"] == 31
    assert ex["dry_spell"] == 12 and ex["fc_rh"] == 24 and ex["fc_wind"] == 17
    assert ex["fc_rain"] == 0 and ex["danger"] == 3


def test_rain_resets_spell_and_missing_fields_stay_nan():
    ex = parse("Today's maximum temperature was 27.0°C (-2.1 from normal). "
               "12 mm rain received in the last 24 hrs. Chance of light showers over higher reaches.")
    assert ex["dry_spell"] == 0 and ex["rain_mm"] == 12 and ex["fc_rain"] == 1
    assert math.isnan(ex["fc_wind"]) and math.isnan(ex["danger"])


def test_field_note_words_and_litter():
    ex = parse("Range officer report - Raipur, week ending 09 May.\n"
               "Field: thick litter layer (6-8 cm) under chir. Crew attended three fires during the week.")
    assert ex["litter"] == 2 and ex["incidents"] == 3


def test_spans_point_at_source_text():
    text = "Forecast Tmax: 36C. Minimum relative humidity expected 21%."
    ex = parse(text)
    pieces = {f: text[a:b] for a, b, f in ex.spans}
    assert pieces["fc_tmax"] == "36C" and pieces["fc_rh"] == "21%"
