"""
Tests for survey number and survey location extraction.

Snippets are taken from real OCR output in section_output/.
"""

from extractor.location_standardizer import (
    PlaceNameStandardizer,
    gazetteer_name,
)
from extractor.normalizer import Normalizer
from extractor.survey_extractor import SurveyExtractor
from extractor.survey_location import SurveyLocationExtractor


def surveys(text):
    return Normalizer.normalize_survey_numbers(SurveyExtractor(text).extract())


def locate(text, numbers=None):
    numbers = numbers or surveys(text)
    results = SurveyLocationExtractor(text, use_llm=False).extract(numbers)
    return {r["survey_number"]: r for r in results}


# ----------------------------------------------------------
# Survey numbers
# ----------------------------------------------------------


def test_list_with_comma_and():
    assert surveys("in Sy.No.85, 50, 37, and 24 in Jamakhandi taluk") == [
        "85", "50", "37", "24",
    ]


def test_list_with_stray_dot():
    assert surveys("Sy.No.76, 77, 78, 81., 82, 83 & 89 of") == [
        "76", "77", "78", "81", "82", "83", "89",
    ]


def test_re_survey_prefix():
    assert surveys("land in RS.No.190 of Savadatti village") == ["190"]


def test_ordinal_not_a_survey():
    assert surveys("Sy.No. 13, 14th km, Hosur Road") == ["13"]


def test_case_number_ignored():
    assert surveys("in O.S.No.158/1984 the land in Survey No.361") == ["361"]


# ----------------------------------------------------------
# Locations
# ----------------------------------------------------------


def test_every_number_in_list_gets_location():
    result = locate(
        "Sy.Nos.4 and 6 of Sathanuru Village, Jala Hobli, "
        "Bangalore North Taluk retrospectively"
    )

    for number in ("4", "6"):
        assert result[number]["village"] == "Sathanuru"
        assert result[number]["hobli"] == "Jala"
        assert result[number]["taluk"] == "Bengaluru North"


def test_location_after_several_mentions():
    result = locate(
        "the lands in Sy. No.52 measuring 1 Acre 34 guntas, Sy. No.61 "
        "measuring 1 Acre 35 guntas and Sy. No.82 measuring 10 Acres, all "
        "situated at Dattagalli Village, Kasaba Hobli, Mysuru Taluk."
    )

    for number in ("52", "61", "82"):
        assert result[number]["village"] == "Dattagalli"


def test_location_before_mention():
    result = locate("Govt. land is available in Savalagi hobli in Sy.No.6")

    assert result["6"]["hobli"] == "Savalagi"


def test_nearest_village_wins():
    result = locate(
        "land in RS. No.190 of Savadatti village, Raibag Taluk or in "
        "Sy. No.73 and 67 of Nasalapur village"
    )

    assert result["190"]["village"] == "Savadatti"
    assert result["73"]["village"] == "Nasalapur"
    assert result["67"]["village"] == "Nasalapur"


def test_label_first_and_combined_taluk_district():
    result = locate(
        "bearing Sy.No.171/1 of Village Nandur-K Taluka and District "
        "Kalaburagi. The respondent"
    )

    assert result["171/1"] == {
        "survey_number": "171/1",
        "village": "Nandur-K",
        "hobli": None,
        "taluk": "Kalaburagi",
        "district": "Kalaburagi",
    }


def test_party_address_is_not_land_location():
    text = (
        "Sri Rajegowda, Aged about 59 Years, R/o Puttanahosalli Village, "
        "Kasaba Hobli, Arakulagudu Taluku, Hasana Dist.\n"
        "The petitioner owned land in Sy. No. 32 of Byagadihalli Village, "
        "Taluk Sakaleshapur, Hasan Dist."
    )

    assert locate(text)["32"]["village"] == "Byagadihalli"


def test_role_words_are_not_places():
    result = locate(
        "Sy.No.9 was inspected by the Taluk Surveyor and Village "
        "Accountant. Sy. No.9 of Lingammanahalli Village, Kadaba Hobli, "
        "Gubbi Taluk."
    )

    assert result["9"]["village"] == "Lingammanahalli"
    assert result["9"]["taluk"] == "Gubbi"


def test_sentence_break_is_not_a_name():
    result = locate(
        "land bearing Sy. No.28 measuring 6 Acres of Attikatti\n\n"
        "Village. There was partition of the land."
    )

    assert result["28"]["village"] == "Attikatti"


def test_subdivision_inherits_parent_location():
    result = locate(
        "Sy.No.33 and 34 of Kenchanahalli Village. Later the site in "
        "Sy.No.33/5 was demolished.",
        ["33", "34", "33/5"],
    )

    assert result["33/5"]["village"] == "Kenchanahalli"


def test_hierarchy_filled_from_elsewhere_in_document():
    result = locate(
        "land in Sy.No.47 of Rayasandra Village.\n"
        "Rayasandra Village, Kasaba Hobli, Kanakapura Taluk is a village."
    )

    assert result["47"]["hobli"] == "Kasaba"
    assert result["47"]["taluk"] == "Kanakapura"


def test_every_survey_number_gets_a_row():
    result = SurveyLocationExtractor(
        "nothing useful here", use_llm=False
    ).extract(["1", "2/3"])

    assert [r["survey_number"] for r in result] == ["1", "2/3"]
    assert all(r["village"] is None for r in result)


# ----------------------------------------------------------
# Gap filling
# ----------------------------------------------------------


def test_district_filled_from_taluk():
    result = locate(
        "Sy.No.47 of Rayasandra Village, Kasaba Hobli, Kanakapura Taluk."
    )

    assert result["47"]["district"] == "Ramanagara"
    assert set(result["47"]) == {
        "survey_number", "village", "hobli", "taluk", "district",
    }


# ----------------------------------------------------------
# Spelling standardisation
# ----------------------------------------------------------


def test_gazetteer_old_names_and_ocr_variants():
    assert gazetteer_name("taluk", "Mysore") == "Mysuru"
    assert gazetteer_name("taluk", "Sakaleshapura") == "Sakleshpur"
    assert gazetteer_name("taluk", "Nelamangata") == "Nelamangala"
    assert gazetteer_name("district", "Baltari") == "Ballari"
    assert gazetteer_name("district", "Hasana") == "Hassan"
    assert gazetteer_name("district", "Tumkur") == "Tumakuru"
    assert gazetteer_name("district", "Bangalore") == "Bengaluru"
    assert gazetteer_name("taluk", "Hedy Mb") is None


def test_village_variants_merged_within_document():
    standardize = PlaceNameStandardizer(
        {"village": ["Lalithadripura"] * 3 + ["Lalithadrapura", "Kuragsd"]}
    )

    assert standardize("village", "Lalithadrapura") == "Lalithadripura"
    assert standardize("village", "NANDUR-K") == "Nandur-K"


def test_distinct_villages_not_merged():
    standardize = PlaceNameStandardizer(
        {"village": ["Chikka Agrahara", "Dodda Agrahara"]}
    )

    assert standardize("village", "Dodda Agrahara") == "Dodda Agrahara"


def test_standardised_names_in_results():
    result = locate(
        "land in Sy.No.76 in Lalithadripura Village, Varuna Hobli, Mysore "
        "Taluk. Sy.No.77 of Lalithadrapura Village, Mysuru Taluk."
    )

    for number in ("76", "77"):
        assert result[number]["village"] == "Lalithadripura"
        assert result[number]["taluk"] == "Mysuru"
