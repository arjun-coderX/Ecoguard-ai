"""Writes the text reports the model has to read.

Two kinds of document per forest range:

* an evening weather bulletin (IMD style) issued on day d with today's
  observations and tomorrow's forecast - roughly a quarter are in Hindi,
  as the Dehradun Met Centre issues both;
* a weekly field note from the range officer: needle litter on the forest
  floor and fire incidents attended.

Phrasing, sentence order and units vary, and fields are sometimes left
out, so the parser cannot just slice fixed positions. The forecast in the
bulletin is the truth for d+1 plus forecast error.
"""
from datetime import date, timedelta

import numpy as np

from .config import CFG

LEVELS_EN = ["low", "moderate", "high", "very high", "extreme"]
LEVELS_HI = ["निम्न", "मध्यम", "उच्च", "अति उच्च", "अत्यधिक"]
RANGE_HI = {"Chakrata": "चकराता", "Kalsi": "कालसी", "Mussoorie": "मसूरी",
            "Raipur": "रायपुर", "Narendranagar": "नरेंद्रनगर"}
DIR_NAME = {"N": "north", "NE": "northeast", "E": "east", "SE": "southeast",
            "S": "south", "SW": "southwest", "W": "west", "NW": "northwest"}
DIR_ADJ = {"N": "northerly", "NE": "north-easterly", "E": "easterly",
           "SE": "south-easterly", "S": "southerly", "SW": "south-westerly",
           "W": "westerly", "NW": "north-westerly"}
LITTER_PHRASES = [
    ["light needle fall so far", "litter still light", "low fuel load on forest floor"],
    ["moderate litter on slopes", "needle fall picking up", "moderate fuel load"],
    ["heavy pine needle accumulation on south slopes",
     "thick litter layer (6-8 cm) under chir", "heavy fuel load, needles everywhere"],
]
NUM_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten"]


def season_date(season, day, cfg=CFG):
    m, d = cfg.season_start_md
    return date(season, m, d) + timedelta(days=int(day))


def danger_level(tmax, rh, wind, spell, litter):
    score = (0.13 * (tmax - 28) - 0.055 * (rh - 30) + 0.04 * (wind - 10)
             + 0.035 * min(spell, 40) + 0.35 * litter)
    return int(np.clip(np.digitize(score, [0.3, 1.2, 2.1, 3.0]), 0, 4))


def _pick(rng, options):
    return options[rng.integers(len(options))]


def _en_bulletin(rng, rname, when, obs, fc):
    head = _pick(rng, [
        f"IMD MC Dehradun | Range forecast: {rname} | Issued {when:%d %b %Y} 17:30 IST",
        f"District weather bulletin - {rname} forest range ({when:%d/%m/%Y})",
        f"{rname.upper()} - evening update, {when:%A %d %B}",
    ])
    s = []
    dep = obs["tmax"] - obs["normal"]
    word = "above" if dep >= 0 else "below"
    s.append(_pick(rng, [
        f"Today's maximum temperature was {obs['tmax']:.1f}°C ({dep:+.1f} from normal).",
        f"Max temp recorded today {obs['tmax']:.1f} deg C, {abs(dep):.1f} {word} normal.",
        f"Observed Tmax {obs['tmax']:.1f}C.",
    ]))
    if obs["rain"] >= 1:
        s.append(_pick(rng, [f"Rainfall recorded today: {obs['rain']:.0f} mm.",
                             f"{obs['rain']:.0f} mm rain received in the last 24 hrs."]))
    elif rng.random() > 0.08:
        n = obs["spell"]
        s.append(_pick(rng, [f"No significant rainfall for the last {n} days.",
                             f"Dry spell enters day {n}.",
                             f"{n} consecutive dry days so far."]))
    if rng.random() > 0.05:
        nxt = (when + timedelta(days=1)).strftime("%A")
        s.append(_pick(rng, [
            f"Tomorrow maximum temperature likely around {fc['tmax']:.0f}°C.",
            f"Max temp expected to touch {fc['tmax']:.0f} deg C tomorrow.",
            f"Forecast Tmax: {fc['tmax']:.0f}C.",
            f"Day temperatures to hover near {fc['tmax']:.0f}°C on {nxt}.",
        ]))
    if rng.random() > 0.10:
        s.append(_pick(rng, [
            f"Minimum relative humidity expected {fc['rh']}%.",
            f"RH likely to drop to {fc['rh']} per cent in the afternoon.",
            f"Afternoon humidity: {fc['rh']}-{fc['rh_max']}%.",
        ]))
    if rng.random() > 0.07:
        g = fc["gust"] > fc["wind"]
        d = fc["dir"]
        s.append(_pick(rng, [
            f"Winds {DIR_ADJ[d]} at {fc['wind']} km/h" + (f", gusting to {fc['gust']} km/h." if g else "."),
            f"Surface wind {d} {fc['wind']} kmph" + (f", gusts {fc['gust']} kmph." if g else "."),
            f"Wind speed around {fc['wind']} km/hr from the {DIR_NAME[d]}" + (f" with gusts up to {fc['gust']} km/hr." if g else "."),
        ]))
    s.append(_pick(rng, [
        "Isolated light rain / thundershowers likely.",
        "Chance of light showers over higher reaches.",
        "Thunderstorm with lightning possible at isolated places.",
    ] if fc["rain"] else [
        "Dry weather likely to prevail.", "No rain expected.", "Mainly clear sky.",
    ]))
    if rng.random() > 0.12:
        lv = LEVELS_EN[fc["danger"]]
        s.append(_pick(rng, [
            f"Forest fire danger: {lv.upper()}.",
            f"FSI fire alert level for {rname}: {lv}.",
            f"Fire danger rating {lv}; field staff advised to remain vigilant.",
        ]))
    body = s[:1] + list(rng.permutation(s[1:-1])) + s[-1:] if len(s) > 2 else s
    return head + "\n" + " ".join(body)


def _hi_bulletin(rng, rname, when, obs, fc):
    rh_name = RANGE_HI[rname]
    s = [f"मौसम बुलेटिन – {rh_name} वन क्षेत्र, {when:%d.%m.%Y}",
         f"आज अधिकतम तापमान {obs['tmax']:.1f}°C रहा।"]
    if obs["rain"] >= 1:
        s.append(f"आज {obs['rain']:.0f} मिमी वर्षा दर्ज।")
    elif rng.random() > 0.08:
        s.append(f"पिछले {obs['spell']} दिनों से कोई वर्षा नहीं।")
    if rng.random() > 0.05:
        s.append(f"कल अधिकतम तापमान लगभग {fc['tmax']:.0f}°C रहने की संभावना।")
    if rng.random() > 0.10:
        s.append(f"न्यूनतम सापेक्ष आर्द्रता {fc['rh']}% तक।")
    if rng.random() > 0.07:
        w = f"हवा {fc['wind']} किमी/घंटा की गति से"
        if fc["gust"] > fc["wind"]:
            w += f", झोंके {fc['gust']} किमी/घंटा तक"
        s.append(w + "।")
    s.append("कहीं-कहीं हल्की वर्षा/गरज के साथ बौछारें संभव।" if fc["rain"]
             else "मौसम शुष्क रहेगा।")
    if rng.random() > 0.12:
        s.append(f"वनाग्नि खतरा स्तर: {LEVELS_HI[fc['danger']]}।")
    return s[0] + "\n" + " ".join(s[1:])


def _field_note(rng, rname, when, litter, incidents):
    lit = _pick(rng, LITTER_PHRASES[litter])
    if incidents == 0:
        inc = _pick(rng, ["No fire incidents this week.", "Nil fire incidents reported."])
    else:
        k = NUM_WORDS[incidents] if incidents <= 10 and rng.random() < 0.4 else str(incidents)
        inc = _pick(rng, [f"{k} fire incidents attended.",
                          f"Crew attended {k} fires during the week.",
                          f"Fires reported: {k}."])
    extra = _pick(rng, ["Fire lines along the forest road cleared.",
                        "Crew of 12 on standby at range office.",
                        "Watchtower staffed from 10am-6pm.",
                        "Villagers briefed on controlled burning ban.", ""])
    return (f"Range officer report - {rname}, week ending {when:%d %b}.\n"
            f"Field: {lit}. {inc} {extra}").strip()


def write_reports(terrain, wx, burning, season, cfg=CFG):
    rng = np.random.default_rng([cfg.seed, season, 4])
    D, R = wx.tmax.shape
    docs = []
    for d in range(D - 1):
        when = season_date(season, d)
        for r, (rname, *_) in enumerate(cfg.ranges):
            obs = dict(tmax=wx.tmax[d, r], normal=wx.tmax_normal[d, r],
                       rain=wx.rain[d, r], spell=wx.dry_spell[d, r])
            t1 = d + 1
            rain_true = wx.rain[t1, r] >= 1
            fc_rain = (rng.random() < 0.8) if rain_true else (rng.random() < 0.08)
            fc = dict(
                tmax=wx.tmax[t1, r] + rng.normal(0, 1.1),
                rh=int(np.clip(wx.rh_min[t1, r] + rng.normal(0, 5), 8, 90)),
                rh_max=int(wx.rh_max[t1, r]),
                wind=int(max(3, wx.wind[t1, r] + rng.normal(0, 2.5))),
                gust=int(wx.gust[t1, r]),
                dir=wx.wind_dir[t1, r],
                rain=bool(fc_rain),
            )
            fc["danger"] = danger_level(fc["tmax"], fc["rh"], fc["wind"],
                                        wx.dry_spell[d, r] + 1, wx.litter[d, r])
            lang = "hi" if rng.random() < 0.25 else "en"
            text = (_hi_bulletin if lang == "hi" else _en_bulletin)(rng, rname, when, obs, fc)
            docs.append(dict(season=season, day=d, range=r, kind="bulletin",
                             lang=lang, text=text,
                             truth=dict(fc_tmax=round(float(fc["tmax"])),
                                        fc_rh=fc["rh"], fc_wind=fc["wind"],
                                        fc_rain=int(fc["rain"]),
                                        dry_spell=int(obs["spell"]) if obs["rain"] < 1 else 0,
                                        danger=fc["danger"])))
            if d % 7 == 6:
                in_range = terrain.range_id == r
                week = burning[d - 6:d + 1][:, in_range].sum()
                incidents = int(min(30, round(week / 2.5)))
                docs.append(dict(season=season, day=d, range=r, kind="field_note",
                                 lang="en",
                                 text=_field_note(rng, rname, when, wx.litter[d, r], incidents),
                                 truth=dict(litter=int(wx.litter[d, r]),
                                            incidents=incidents)))
    return docs
