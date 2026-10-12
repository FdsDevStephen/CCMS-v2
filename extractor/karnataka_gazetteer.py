"""
Karnataka districts and taluks (official spellings), plus old names and
common variants seen in court documents.

Used by extractor.location_standardizer to standardise taluk / district
names ("Mysore" -> "Mysuru", "Hasana" -> "Hassan").
"""

from __future__ import annotations

# District -> taluks.
TALUKS_BY_DISTRICT: dict[str, list[str]] = {
    "Bagalkot": [
        "Badami", "Bagalkot", "Bilagi", "Guledgudda", "Hungund", "Ilkal",
        "Jamkhandi", "Mudhol", "Rabkavi Banhatti", "Terdal",
    ],
    "Ballari": ["Ballari", "Kampli", "Kurugodu", "Sandur", "Siruguppa"],
    "Belagavi": [
        "Athani", "Bailhongal", "Belagavi", "Chikkodi", "Gokak", "Hukkeri",
        "Kagwad", "Khanapur", "Kittur", "Mudalgi", "Nippani", "Ramdurg",
        "Raybag", "Savadatti", "Yaragatti",
    ],
    "Bengaluru Urban": [
        "Anekal", "Bengaluru East", "Bengaluru North", "Bengaluru South",
        "Yelahanka",
    ],
    "Bengaluru Rural": [
        "Devanahalli", "Doddaballapura", "Hoskote", "Nelamangala",
    ],
    "Bidar": [
        "Aurad", "Basavakalyan", "Bhalki", "Bidar", "Chitguppa", "Hulsoor",
        "Humnabad", "Kamalnagar",
    ],
    "Chamarajanagar": [
        "Chamarajanagar", "Gundlupet", "Hanur", "Kollegal", "Yelandur",
    ],
    "Chikkaballapur": [
        "Bagepalli", "Cheluru", "Chikkaballapur", "Chintamani",
        "Gauribidanur", "Gudibanda", "Manchenahalli", "Sidlaghatta",
    ],
    "Chikkamagaluru": [
        "Ajjampura", "Chikkamagaluru", "Kadur", "Kalasa", "Koppa",
        "Mudigere", "Narasimharajapura", "Sringeri", "Tarikere",
    ],
    "Chitradurga": [
        "Challakere", "Chitradurga", "Hiriyur", "Holalkere", "Hosadurga",
        "Molakalmuru",
    ],
    "Dakshina Kannada": [
        "Bantwal", "Belthangady", "Kadaba", "Mangaluru", "Moodbidri",
        "Mulki", "Puttur", "Sullia", "Ullal",
    ],
    "Davanagere": [
        "Channagiri", "Davanagere", "Harihar", "Honnali", "Jagalur",
        "Nyamathi",
    ],
    "Dharwad": [
        "Alnavar", "Annigeri", "Dharwad", "Hubballi", "Kalghatgi",
        "Kundgol", "Navalgund",
    ],
    "Gadag": [
        "Gadag", "Gajendragad", "Lakshmeshwar", "Mundargi", "Nargund",
        "Ron", "Shirahatti",
    ],
    "Hassan": [
        "Alur", "Arkalgud", "Arsikere", "Belur", "Channarayapatna",
        "Hassan", "Holenarasipur", "Sakleshpur",
    ],
    "Haveri": [
        "Byadgi", "Hanagal", "Haveri", "Hirekerur", "Ranebennur",
        "Rattihalli", "Savanur", "Shiggaon",
    ],
    "Kalaburagi": [
        "Afzalpur", "Aland", "Chincholi", "Chittapur", "Jevargi",
        "Kalaburagi", "Kalagi", "Kamalapur", "Sedam", "Shahabad", "Yadrami",
    ],
    "Kodagu": [
        "Kushalnagar", "Madikeri", "Ponnampet", "Somwarpet", "Virajpet",
    ],
    "Kolar": [
        "Bangarapet", "Kolar", "Kolar Gold Fields", "Malur", "Mulbagal",
        "Srinivaspur",
    ],
    "Koppal": [
        "Gangavathi", "Kanakagiri", "Karatagi", "Koppal", "Kukanoor",
        "Kushtagi", "Yelburga",
    ],
    "Mandya": [
        "Krishnarajpet", "Maddur", "Malavalli", "Mandya", "Nagamangala",
        "Pandavapura", "Srirangapatna",
    ],
    "Mysuru": [
        "Heggadadevanakote", "Hunsur", "Krishnarajanagara", "Mysuru",
        "Nanjangud", "Piriyapatna", "Saligrama", "Saragur",
        "Tirumakudalu Narasipura",
    ],
    "Raichur": [
        "Devadurga", "Lingasugur", "Manvi", "Maski", "Raichur",
        "Sindhanur", "Sirwar",
    ],
    "Ramanagara": [
        "Channapatna", "Harohalli", "Kanakapura", "Magadi", "Ramanagara",
    ],
    "Shivamogga": [
        "Bhadravati", "Hosanagara", "Sagar", "Shikaripur", "Shivamogga",
        "Soraba", "Thirthahalli",
    ],
    "Tumakuru": [
        "Chikkanayakanahalli", "Gubbi", "Koratagere", "Kunigal",
        "Madhugiri", "Pavagada", "Sira", "Tiptur", "Tumakuru", "Turuvekere",
    ],
    "Udupi": [
        "Brahmavar", "Byndoor", "Hebri", "Kapu", "Karkala", "Kundapura",
        "Udupi",
    ],
    "Uttara Kannada": [
        "Ankola", "Bhatkal", "Dandeli", "Haliyal", "Honnavar", "Joida",
        "Karwar", "Kumta", "Mundgod", "Siddapur", "Sirsi", "Yellapur",
    ],
    "Vijayanagara": [
        "Hagaribommanahalli", "Harapanahalli", "Hosapete",
        "Hoovina Hadagali", "Kotturu", "Kudligi",
    ],
    "Vijayapura": [
        "Alamel", "Babaleshwar", "Basavana Bagewadi", "Chadachan",
        "Devara Hipparagi", "Indi", "Kolhar", "Muddebihal", "Nidagundi",
        "Sindagi", "Talikoti", "Tikota", "Vijayapura",
    ],
    "Yadgir": [
        "Gurmitkal", "Hunsagi", "Shahapur", "Shorapur", "Vadagera",
        "Yadgir",
    ],
}

# Old names / variants -> official district name.
DISTRICT_ALIASES: dict[str, str] = {
    "Bagalkote": "Bagalkot",
    "Bellary": "Ballari",
    "Belgaum": "Belagavi",
    # Old "Bangalore District" covered today's Urban and Rural districts,
    # so only the spelling is modernised.
    "Bangalore": "Bengaluru",
    "Bengaluru": "Bengaluru",
    "Bangalore Urban": "Bengaluru Urban",
    "Bangalore Rural": "Bengaluru Rural",
    "Chamarajanagara": "Chamarajanagar",
    "Chikballapur": "Chikkaballapur",
    "Chikkaballapura": "Chikkaballapur",
    "Chikmagalur": "Chikkamagaluru",
    "Chikkamagalur": "Chikkamagaluru",
    "Chikkamangalur": "Chikkamagaluru",
    "Chickmagalur": "Chikkamagaluru",
    "Davangere": "Davanagere",
    "Gulbarga": "Kalaburagi",
    "Hasan": "Hassan",
    "Hasana": "Hassan",
    "Coorg": "Kodagu",
    "Mysore": "Mysuru",
    "Ramanagaram": "Ramanagara",
    "Shimoga": "Shivamogga",
    "Shivamogga": "Shivamogga",
    "Tumkur": "Tumakuru",
    "Tumkuru": "Tumakuru",
    "Bijapur": "Vijayapura",
    "Yadagiri": "Yadgir",
    "South Kanara": "Dakshina Kannada",
    "North Kanara": "Uttara Kannada",
    "Karwar": "Uttara Kannada",
    "Mangalore": "Dakshina Kannada",
    "Mangaluru": "Dakshina Kannada",
}

# Old names / variants -> official taluk name.
TALUK_ALIASES: dict[str, str] = {
    "Bangalore North": "Bengaluru North",
    "Bangalore South": "Bengaluru South",
    "Bangalore East": "Bengaluru East",
    "Mysore": "Mysuru",
    "Tumkur": "Tumakuru",
    "Shimoga": "Shivamogga",
    "Bellary": "Ballari",
    "Belgaum": "Belagavi",
    "Gulbarga": "Kalaburagi",
    "Bijapur": "Vijayapura",
    "Davangere": "Davanagere",
    "Hubli": "Hubballi",
    "Mangalore": "Mangaluru",
    "Hospet": "Hosapete",
    "Hosapet": "Hosapete",
    "Bagalkote": "Bagalkot",
    "Jamakhandi": "Jamkhandi",
    "Raibag": "Raybag",
    "Saundatti": "Savadatti",
    "Arakalagudu": "Arkalgud",
    "Arakalgud": "Arkalgud",
    "Arkalgudu": "Arkalgud",
    "Arakulagudu": "Arkalgud",
    "Aluru": "Alur",
    "Aalur": "Alur",
    "Beluru": "Belur",
    "Sakaleshpur": "Sakleshpur",
    "Sakaleshapura": "Sakleshpur",
    "Sakaleshpura": "Sakleshpur",
    "Sakleshpura": "Sakleshpur",
    "Holenarasipura": "Holenarasipur",
    "Hole Narasipur": "Holenarasipur",
    "Channarayapattana": "Channarayapatna",
    "Hasan": "Hassan",
    "Hasana": "Hassan",
    "Periyapatna": "Piriyapatna",
    "Periyapattana": "Piriyapatna",
    "H.D. Kote": "Heggadadevanakote",
    "H D Kote": "Heggadadevanakote",
    "K.R. Nagar": "Krishnarajanagara",
    "K.R. Pet": "Krishnarajpet",
    "K R Pet": "Krishnarajpet",
    "T. Narasipura": "Tirumakudalu Narasipura",
    "T.Narasipura": "Tirumakudalu Narasipura",
    "Belthangadi": "Belthangady",
    "Bantwala": "Bantwal",
    "Joyda": "Joida",
    "Mundgodu": "Mundgod",
    "Ramanagaram": "Ramanagara",
    "Srinivasapura": "Srinivaspur",
    "K.G.F.": "Kolar Gold Fields",
    "KGF": "Kolar Gold Fields",
    "Kundapur": "Kundapura",
    "Brahmavara": "Brahmavar",
    "Baindur": "Byndoor",
    "Shirhatti": "Shirahatti",
    "Mundaragi": "Mundargi",
    "Lingsugur": "Lingasugur",
    "Lingasugur": "Lingasugur",
    "Sorab": "Soraba",
    "Shikaripura": "Shikaripur",
    "Sagara": "Sagar",
    "Tirthahalli": "Thirthahalli",
    "Bhadravathi": "Bhadravati",
    "Chikmagalur": "Chikkamagaluru",
    "Chikkamangalur": "Chikkamagaluru",
    "N.R. Pura": "Narasimharajapura",
    "C.N. Halli": "Chikkanayakanahalli",
    "Madhugiri": "Madhugiri",
    "Gangavati": "Gangavathi",
    "Yelbarga": "Yelburga",
    "Hangal": "Hanagal",
    "Ranibennur": "Ranebennur",
    "Muddebihala": "Muddebihal",
    "Sindgi": "Sindagi",
    "B. Bagewadi": "Basavana Bagewadi",
    "Hadagali": "Hoovina Hadagali",
    "H.B. Halli": "Hagaribommanahalli",
    "Doddaballapur": "Doddaballapura",
    "Hosakote": "Hoskote",
    "Somvarpet": "Somwarpet",
    "Virajpete": "Virajpet",
    "Kollegala": "Kollegal",
    "Gundlupete": "Gundlupet",
    "Nanjangudu": "Nanjangud",
    "Srirangapattana": "Srirangapatna",
    "Malavalli": "Malavalli",
    "Kanakapura": "Kanakapura",
    "Hiriyuru": "Hiriyur",
    "Challakeri": "Challakere",
    "Honnalli": "Honnali",
    "Harapanahalli": "Harapanahalli",
}

# Taluk -> district (for filling a missing district from a known taluk).
DISTRICT_OF_TALUK: dict[str, str] = {
    taluk: district
    for district, taluks in TALUKS_BY_DISTRICT.items()
    for taluk in taluks
}

DISTRICTS: list[str] = list(TALUKS_BY_DISTRICT)
TALUKS: list[str] = list(DISTRICT_OF_TALUK)
