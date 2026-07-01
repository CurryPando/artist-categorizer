import pandas as pd
from transformers import BertTokenizer

def ingest_data(file_path: str, artists: set[str]) -> pd.DataFrame:
    print('loading data')
    raw_data = pd.read_csv(file_path)
    raw_data = raw_data[raw_data['artist'].isin(artists)][['artist', 'song', 'lyric']]
    raw_data.dropna(inplace=True)
    for column in ['artist', 'song', 'lyric']:
        if not pd.api.types.is_string_dtype(raw_data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")
    return raw_data

def chunk_lyric_dataframe(df: pd.DataFrame, tokenizer: BertTokenizer, min_tokens: int = 64, max_tokens: int = 128, overlap: int = 16) -> tuple[list[str], list[str]]:
    """
    Groups a lyric dataframe by song, aggregates the lines, and splits them into 
    overlapping chunks based on token counts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'artist', 'song', and 'lyric' columns.
    tokenizer : BertTokenizer
        The Hugging Face tokenizer to use.
    min_tokens : int
        The minimum number of tokens a chunk must have to be kept (prevents tiny trailing chunks).
    max_tokens : int
        The maximum window size for each chunk.
    overlap : int
        The number of tokens to overlap between consecutive chunks.
    """
    stride = max_tokens - overlap
    
    if stride <= 0:
        raise ValueError("Overlap must be strictly less than max_tokens.")
        
    chunked_records = []
    
    for (artist, _), group in df.groupby(['artist', 'song']):
        # Combine all lines for this song into a single string, space-separated
        full_song_text = " ".join(group['lyric'].astype(str).str.strip().tolist())
        
        # Tokenize the entire song into token IDs (without padding/truncation yet)
        tokenized = tokenizer(full_song_text, add_special_tokens=False)
        input_ids = tokenized['input_ids']
        
        total_tokens = len(input_ids)
        
        # If the whole song is shorter than the minimum token limit, keep it as one chunk
        if total_tokens <= max_tokens:
            if total_tokens >= min_tokens:
                chunk_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                chunked_records.append({'artist': artist, 'chunk_lyric': chunk_text})
            continue
            
        # Sliding window logic
        start_idx = 0
        while start_idx < total_tokens:
            end_idx = start_idx + max_tokens
            chunk_ids = input_ids[start_idx:end_idx]
            
            # Check if the trailing chunk meets the minimum token size requirement
            if len(chunk_ids) < min_tokens:
                break
                
            # Decode token IDs back into actual text strings
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunked_records.append({'artist': artist, 'chunk_lyric': chunk_text})
            
            # Slide the window forward
            start_idx += stride

    artists_out = [record['artist'] for record in chunked_records]
    lyrics_out = [record['chunk_lyric'] for record in chunked_records]
    return artists_out, lyrics_out