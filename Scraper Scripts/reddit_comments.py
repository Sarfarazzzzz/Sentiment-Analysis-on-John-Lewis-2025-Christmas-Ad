import pandas as pd
import praw

# --- CONFIGURATION ---
POST_URL = "https://www.reddit.com/r/AskUK/comments/1oq0vpl/what_are_your_thoughts_on_the_john_lewis/"
CLIENT_ID = "CREATE AND USE YOUR OWN ID"
CLIENT_SECRET = "CREATE AND USE YOUR OWN ID"
USER_AGENT = "CREATE AND USE YOUR OWN ID"
# ---------------------

comments = []

try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    submission = reddit.submission(url=POST_URL)

    print(f"Fetching comments from Reddit ({POST_URL})...")

    submission.comments.replace_more(limit=None)

    for comment in submission.comments.list():
        comments.append({
            'source': 'Reddit',
            'author': str(comment.author),
            'comment': comment.body,
            'published_at': pd.to_datetime(comment.created_utc, unit='s'),
            'like_count': comment.score
        })

    print(f"Successfully collected {len(comments)} Reddit comments.")

    # Save to a DataFrame
    df_reddit = pd.DataFrame(comments)
    df_reddit.to_csv("reddit_comments.csv", index=False, encoding='utf-8')
    print("Saved to reddit_comments.csv")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your Reddit credentials and username are correct.")
