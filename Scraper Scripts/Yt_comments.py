import pandas as pd
from googleapiclient.discovery import build

# --- CONFIGURATION ---
VIDEO_ID = "CREATE AND USE YOUR OWN ID"
API_KEY = "CREATE AND USE YOUR OWN KEY"
# ---------------------

comments = []

try:
    # Build the service object
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Initial request
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=VIDEO_ID,
        maxResults=100,
        textFormat="plainText"
    )

    print(f"Fetching comments from YouTube (Video ID: {VIDEO_ID})...")

    # Loop through all pages of comments
    while request:
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'source': 'YouTube',
                'author': comment['authorDisplayName'],
                'comment': comment['textDisplay'],
                'published_at': comment['publishedAt'],
                'like_count': comment['likeCount']
            })

        # Check if there is a next page
        request = youtube.commentThreads().list_next(request, response)

    print(f"Successfully collected {len(comments)} YouTube comments.")

    # Save to a DataFrame
    df_youtube = pd.DataFrame(comments)
    df_youtube.to_csv("youtube_comments.csv", index=False, encoding='utf-8')
    print("Saved to youtube_comments.csv")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your API Key is correct and 'YouTube Data API v3' is enabled.")
