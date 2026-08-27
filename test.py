import pandas as pd

ARTISTS = {
    "Drake",
    "Eminem",
    "Kanye West",
    "Kendrick Lamar",
    "J. Cole",
    "XXXTENTACION",
    "Travis Scott",
    "Lil Wayne",
    "JAY-Z",
    "Juice WRLD",
    "Future",
    "Nicki Minaj",
    "Tyler, The Creator",
    "Lil Uzi Vert",
    "A$AP Rocky",
    "Migos",
    "Cardi B",
    "2Pac",
    "Young Thug",
    "Mac Miller",
    "YoungBoy Never Broke Again",
    "Chance the Rapper",
    "Chief Keef",
    "The Notorious B.I.G.",
    "Nas",
    "Playboi Carti",
    "Fetty Wap",
    "Kid Cudi",
    "50 Cent",
    "ScHoolboy Q",
    "A Tribe Called Quest",
    "Common",
    "2 Chainz",
    "Gucci Mane",
    "Earl Sweatshirt",
    "Pop Smoke",
    "Pusha T",
    "OutKast",
}

data = pd.read_csv("train_df.csv", encoding="utf-8")

# print artists ordered by total lyric length
artist_lyrics = data.groupby('artist')['lyrics'].apply(lambda x: ' '.join(x))
artist_lyrics_length = artist_lyrics.apply(len)
artist_lyrics_length = artist_lyrics_length.sort_values(ascending=False)
for artist, length in artist_lyrics_length.items():
    print(f"{artist}: {length}")

CUT = {
    "Cardi B", # 0.05
    "Earl Sweatshirt", # 0.19
    "Pusha T", # 0.08
    "Fetty Wap", # 0.29
    "Chance the Rapper", # 0.09
    "XXXTENTACION", # 0.21
    "Pop Smoke", # 0.28
    "A Tribe Called Quest", # 0.21
    "ScHoolboy Q", # 0.16
}

CUT_2 = {
    "The Notorious B.I.G.", # 0.17
    "OutKast", # 0.18
}

CUT_3 = {
    "A$AP Rocky" # 0.12
}