"""Neutral appearance-attribute vocabulary.

Design rule (PS02 "Strong" requirement): appearance attributes are stored and
rendered as *neutral descriptive traits*. Nationality / region labels live in a
separate ``region_context`` field that is used ONLY for coverage bookkeeping and
is NEVER injected into a generation prompt. ``prompt.build_prompt`` therefore
reads from ``AvatarSpec.appearance`` alone.

This keeps the system from encoding "nationality => look" stereotypes while
still letting us report geographic/cultural coverage of a test matrix.
"""

from __future__ import annotations

# Controlled vocabularies. Values are neutral, descriptive and non-ethnic.
AGE_BANDS = ("young-adult", "adult", "middle-aged", "senior")

PRESENTATIONS = ("masculine", "feminine", "androgynous")

# Monk Skin Tone scale (10 tones, open research scale) - referenced by tone id
# so we never say "X nationality has Y skin".
SKIN_TONES = tuple(f"mst-{i}" for i in range(1, 11))

SKIN_TONE_PHRASES = {
    "mst-1": "very light skin",
    "mst-2": "light skin",
    "mst-3": "light-medium skin",
    "mst-4": "medium-light skin",
    "mst-5": "medium skin",
    "mst-6": "medium-tan skin",
    "mst-7": "tan-brown skin",
    "mst-8": "brown skin",
    "mst-9": "deep brown skin",
    "mst-10": "deep skin",
}

HAIR_LENGTHS = ("shaved", "short", "chin-length", "shoulder-length", "long")
HAIR_TEXTURES = ("straight", "wavy", "curly", "coily", "locs", "none")
HAIR_COLOURS = ("black", "dark-brown", "brown", "auburn", "blonde", "grey", "white", "none")

ATTIRE = (
    "plain-crewneck",
    "collared-shirt",
    "business-blazer",
    "knit-sweater",
    "lab-coat",
    "high-visibility-jacket",
    "embroidered-tunic",
    "draped-shawl",
    "headwrap",
    "turban",
    "hijab",
    "kippah",
)

BACKGROUNDS = (
    "neutral-grey-studio",
    "soft-white-studio",
    "warm-beige-studio",
    "deep-teal-studio",
    "office-bokeh",
    "outdoor-daylight-bokeh",
    "bookshelf-bokeh",
)

POSES = (
    "front-facing-headshot",
    "three-quarter-headshot",
    "profile-headshot",
    "shoulders-up-relaxed",
    "seated-upper-body",
)

EXPRESSIONS = ("neutral", "slight-smile", "smiling", "thoughtful")

EYEWEAR = ("none", "thin-frame-glasses", "thick-frame-glasses")
FACIAL_HAIR = ("none", "stubble", "short-beard", "full-beard", "moustache")

# Phrase templates used to turn controlled values into prompt fragments.
PHRASES = {
    "age_band": {
        "young-adult": "a person in their twenties",
        "adult": "a person in their thirties",
        "middle-aged": "a person in their late forties",
        "senior": "a person in their late sixties",
    },
    "presentation": {
        "masculine": "masculine presenting",
        "feminine": "feminine presenting",
        "androgynous": "androgynous presenting",
    },
    "hair_length": {
        "shaved": "shaved head",
        "short": "short hair",
        "chin-length": "chin-length hair",
        "shoulder-length": "shoulder-length hair",
        "long": "long hair",
    },
    "hair_texture": {
        "straight": "straight",
        "wavy": "wavy",
        "curly": "curly",
        "coily": "tightly coiled",
        "locs": "in locs",
        "none": "",
    },
    "attire": {
        "plain-crewneck": "wearing a plain crewneck t-shirt",
        "collared-shirt": "wearing a collared button-up shirt",
        "business-blazer": "wearing a tailored business blazer",
        "knit-sweater": "wearing a knit sweater",
        "lab-coat": "wearing a white lab coat",
        "high-visibility-jacket": "wearing a high-visibility work jacket",
        "embroidered-tunic": "wearing an embroidered tunic",
        "draped-shawl": "wearing a draped shawl over one shoulder",
        "headwrap": "wearing a fabric headwrap",
        "turban": "wearing a turban",
        "hijab": "wearing a hijab",
        "kippah": "wearing a kippah",
    },
    "background": {
        "neutral-grey-studio": "plain neutral grey studio backdrop",
        "soft-white-studio": "soft white studio backdrop",
        "warm-beige-studio": "warm beige studio backdrop",
        "deep-teal-studio": "deep teal studio backdrop",
        "office-bokeh": "blurred office interior background",
        "outdoor-daylight-bokeh": "blurred outdoor daylight background",
        "bookshelf-bokeh": "blurred bookshelf background",
    },
    "pose": {
        "front-facing-headshot": "front-facing headshot, centred",
        "three-quarter-headshot": "three-quarter view headshot",
        "profile-headshot": "side profile headshot",
        "shoulders-up-relaxed": "relaxed head-and-shoulders portrait",
        "seated-upper-body": "seated upper-body portrait",
    },
    "expression": {
        "neutral": "neutral expression",
        "slight-smile": "a slight closed-mouth smile",
        "smiling": "a warm smile",
        "thoughtful": "a thoughtful expression",
    },
    "eyewear": {
        "none": "",
        "thin-frame-glasses": "wearing thin-frame glasses",
        "thick-frame-glasses": "wearing thick-frame glasses",
    },
    "facial_hair": {
        "none": "",
        "stubble": "light stubble",
        "short-beard": "a short trimmed beard",
        "full-beard": "a full beard",
        "moustache": "a moustache",
    },
}

# Attributes that the controlled-change test matrix may vary one at a time.
CONTROLLED_ATTRIBUTES = (
    "age_band",
    "hair_length",
    "attire",
    "background",
    "eyewear",
    "expression",
    "skin_tone",
    "pose",
)

VOCABULARIES = {
    "age_band": AGE_BANDS,
    "presentation": PRESENTATIONS,
    "skin_tone": SKIN_TONES,
    "hair_length": HAIR_LENGTHS,
    "hair_texture": HAIR_TEXTURES,
    "hair_colour": HAIR_COLOURS,
    "attire": ATTIRE,
    "background": BACKGROUNDS,
    "pose": POSES,
    "expression": EXPRESSIONS,
    "eyewear": EYEWEAR,
    "facial_hair": FACIAL_HAIR,
}
