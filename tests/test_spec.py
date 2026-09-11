"""Unit tests: the input contract must fail loudly on bad specs."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from openavatar.spec import AvatarSpec, load_specs


def _base(**kw) -> dict:
    d = {"spec_id": "t1"}
    d.update(kw)
    return d


def test_defaults_are_valid():
    s = AvatarSpec.model_validate(_base())
    assert s.appearance.age_band == "adult"
    assert s.render.width == 384
    assert s.render.aspect_ratio == "1:1"


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        AvatarSpec.model_validate(_base(nationality="Indian"))


def test_unknown_attribute_value_is_rejected():
    with pytest.raises(ValidationError):
        AvatarSpec.model_validate(_base(appearance={"attire": "spacesuit"}))


def test_skin_tone_must_be_on_the_scale():
    with pytest.raises(ValidationError) as e:
        AvatarSpec.model_validate(_base(appearance={"skin_tone": "dark"}))
    assert "skin_tone" in str(e.value)


def test_shaved_head_requires_no_hair_texture():
    with pytest.raises(ValidationError):
        AvatarSpec.model_validate(_base(appearance={"hair_length": "shaved", "hair_texture": "curly"}))
    AvatarSpec.model_validate(_base(appearance={"hair_length": "shaved", "hair_texture": "none"}))


def test_dimensions_must_be_multiples_of_eight():
    with pytest.raises(ValidationError):
        AvatarSpec.model_validate(_base(render={"width": 385, "height": 384}))


def test_aspect_ratio_is_derived():
    s = AvatarSpec.model_validate(_base(render={"width": 512, "height": 384}))
    assert s.render.aspect_ratio == "4:3"


def test_individual_intent_requires_references():
    with pytest.raises(ValidationError):
        AvatarSpec.model_validate(_base(intent="individual"))


def test_spec_id_rejects_path_separators():
    with pytest.raises(ValidationError):
        AvatarSpec.model_validate(_base(spec_id="../escape"))


def test_load_specs_rejects_duplicate_ids(tmp_path):
    for name in ("one.json", "two.json"):
        (tmp_path / name).write_text(json.dumps({"spec_id": "dup"}))
    with pytest.raises(ValueError, match="duplicate spec_id"):
        load_specs(tmp_path)


def test_load_specs_errors_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_specs(tmp_path)
